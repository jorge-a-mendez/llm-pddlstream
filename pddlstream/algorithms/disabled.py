import time

from pddlstream.algorithms.common import add_facts, add_certified, is_instance_ready, UNKNOWN_EVALUATION
from pddlstream.algorithms.algorithm import remove_blocked
from pddlstream.language.constants import OptPlan
from pddlstream.language.function import FunctionResult
from pddlstream.language.stream import StreamResult
from pddlstream.language.conversion import is_plan, transform_action_args, replace_expression, pddl_from_obj_plan, \
    pddl_list_from_expression
from pddlstream.utils import INF, safe_zip, apply_mapping, flatten, elapsed_time

# TODO: disabled isn't quite like complexity. Stream instances below the complexity threshold might be called again
# Well actually, if this was true wouldn't it have already been sampled on a lower level?

def update_bindings(bindings, opt_result, result):
    if not isinstance(result, StreamResult):
        return bindings
    new_bindings = bindings.copy()
    for opt, obj in safe_zip(opt_result.output_objects, result.output_objects):
        assert new_bindings.get(opt, obj) == obj  # TODO: return failure if conflicting bindings
        new_bindings[opt] = obj
    return new_bindings

def update_cost(cost, opt_result, result):
    # TODO: recompute optimistic costs to attempt to produce a tighter bound
    if type(result) is not FunctionResult:
        return cost
    return cost + (result.value - opt_result.value)

def bind_action_plan(opt_plan, mapping):
    fn = lambda o: mapping.get(o, o)
    new_action_plan = [transform_action_args(action, fn)
                       for action in opt_plan.action_plan]
    new_preimage_facts = frozenset(replace_expression(fact, fn)
                                   for fact in opt_plan.preimage_facts)
    return OptPlan(new_action_plan, new_preimage_facts)

def get_free_objects(stream_plan):
    return set(flatten(result.output_objects for result in stream_plan
                       if isinstance(result, StreamResult)))



##################################################

def push_disabled(instantiator, disabled):
    for instance in list(disabled):
        if instance.enumerated:
            disabled.remove(instance)
        else:
            # TODO: only add if not already queued
            instantiator.push_instance(instance)

def reenable_disabled(evaluations, domain, disabled):
    for instance in disabled:
        instance.enable(evaluations, domain)
    disabled.clear()

def process_instance(store, domain, instance, disable=False):
    if instance.enumerated:
        return [], []
    start_time = time.time()
    new_results, new_facts = instance.next_results(verbose=store.verbose)
    store.sample_time += elapsed_time(start_time)

    evaluations = store.evaluations
    if disable:
        instance.disable(evaluations, domain)
    for result in new_results:
        #add_certified(evaluations, result)  # TODO: only add if the fact is actually new?
        complexity = INF if (not disable or result.external.is_special) else \
            result.compute_complexity(evaluations)
        add_facts(evaluations, result.get_certified(), result=result, complexity=complexity)
    if disable:
        remove_blocked(evaluations, domain, instance, new_results)
    add_facts(evaluations, new_facts, result=UNKNOWN_EVALUATION, complexity=0) # TODO: record the instance
    return new_results, new_facts

##################################################

def process_stream_plan_backtracking(store, domain, disabled, stream_plan, action_plan, cost,
                        bind=True, max_failures=10):
    # JMM: trying to implement SESAME using the PDDLStream way of doing things
    if not is_plan(stream_plan):
        return
    if not stream_plan:
        store.add_plan(action_plan, cost)
        return
    stream_plan = [result for result in stream_plan if result.optimistic]
    free_objects = get_free_objects(stream_plan)
    bindings = {}
    bound_plan = []
    num_wild = 0
    
    num_tries = [0 for _ in range(len(stream_plan))]  # how many times we've tried each stream
    bindings_added = [tuple() for _ in range(len(stream_plan))]
    idx = 0
    highest_idx = 0
    while not store.is_timeout() and idx >= 0 and idx < len(stream_plan):
        opt_result = stream_plan[idx]
        opt_inputs = [inp for inp in opt_result.instance.input_objects if inp in free_objects]
        if (not bind and opt_inputs) or not all(inp in bindings for inp in opt_inputs):
            # JMM: Means I need some input for which I don't have a binding -- should never happen, I think
            raise NotImplementedError
            break
        bound_result = opt_result.remap_inputs(bindings)
        bound_instance = bound_result.instance
        if num_tries[idx] == 0 and bound_instance.enumerated:# and bound_instance._generator.max_calls == 1:
            # JMM: not sure this works -- I'm just assuming this was a test that always passes
            # JMM: had to do this to prevent getting stuck backtracking on test streams that are being repeatedly evaluated
            # JMM: Maybe I can only do this for ones that already have bindings_added (otherwise they failed in the past, so I need to do something to get them to be retried)
            if len(bindings_added[idx]) == 3:
                print(f"Skipping sample at index {idx} / {len(stream_plan)} for {opt_result}, at try {num_tries[idx]} [{bound_instance._generator.max_calls if not isinstance(opt_result, FunctionResult) and hasattr(bound_instance._generator, 'max_calls') else 'inf'}]")
                bound_result, new_result, idx_opt_result = bindings_added[idx]
                bound_plan.append(new_result)
                bindings = update_bindings(bindings, bound_result, bound_plan[-1])
                cost = update_cost(cost, idx_opt_result, bound_plan[-1])
                num_tries[idx] += 1
                idx += 1
                assert idx <= highest_idx, f"SESAME error: {idx} > {highest_idx} when we already sampled {idx}"
                continue

        if num_tries[idx] >= max_failures or bound_instance.enumerated or not is_instance_ready(store.evaluations, bound_instance):
            # JMM: Means we can't sample the current thing. We should reset it 
            # JMM: It's a bit trickier because streams sometimes are tests. For tests, if they are applied to not-yet-sampled stuff, they 
            # always return true and then they become enumerated, meaning that we can never re-query them. Need to reset those.
            print(f"Backtracking at index {idx} for {opt_result}, with {num_tries[idx]} tries, enumerated= {bound_instance.enumerated}, ready={is_instance_ready(store.evaluations, bound_instance)} [{bound_instance._generator.max_calls if not isinstance(opt_result, FunctionResult) and hasattr(bound_instance._generator, 'max_calls') else 'inf'}]")
            num_tries[idx] = 0
            idx -= 1
            bound_plan = bound_plan[:-1]
            if len(bindings_added[idx]) == 3:
                bound_result, new_result, idx_opt_result = bindings_added[idx]
                if isinstance(new_result, StreamResult):
                    for binding in bound_result.output_objects:
                        if binding in bindings:
                            del bindings[binding]
                if isinstance(new_result, FunctionResult):
                    cost = cost - new_result.value + idx_opt_result.value
            continue
        new_results, new_facts = process_instance(store, domain, bound_instance)
        num_tries[idx] += 1

        assert len(new_facts) == 0, f'SESAME error: {len(new_facts)} != 0'

        for new_result in new_results:
            if new_result.is_successful():
                # bound_plan.append(new_results[0])
                bound_plan.append(new_result)
                bindings = update_bindings(bindings, bound_result, bound_plan[-1])
                bindings_added[idx] = (bound_result, bound_plan[-1], opt_result)
                cost = update_cost(cost, opt_result, bound_plan[-1])
                break
        else:
            # JMM: Means we didn't find a successful result, so we need to try again
            continue
        # JMM: We found a successful result, so we need to try the next stream
        print(f"Found sample at index {idx} / {len(stream_plan)} for {opt_result}, at try {num_tries[idx]} [{bound_instance._generator.max_calls if not isinstance(opt_result, FunctionResult) and hasattr(bound_instance._generator, 'max_calls') else 'inf'}]")
        idx += 1
        highest_idx = max(highest_idx, idx)
    
    if store.is_timeout():
        return
    assert (len(bound_plan) == len(stream_plan)) == (idx == len(stream_plan)), f'SESAME error: {len(bound_plan)} ?= {len(stream_plan)} but {idx} ?= {len(stream_plan)}'
    if len(stream_plan) == len(bound_plan):
        print("Checking cost computation...")
        print("Cost:", cost)
        for result in bound_plan:
            if isinstance(result, FunctionResult):
                print(result, result.value)
        store.add_plan(bind_action_plan(action_plan, bindings), cost)
    elif stream_plan[highest_idx].external.name == "obj-inv-visible":
        new_preconditions = {
            'calibrate': '(NotCausesFailure ?v ?o)',
            'take_image': '(NotCausesFailure ?v ?o)'
        }
        new_effects = {}
        failed_stream = stream_plan[highest_idx]
        remove_init_atom = ('NotCausesFailure', *(failed_stream.instance.input_objects))
        infeasible_action = None
        for action in action_plan.action_plan:
            print(action.name, failed_stream.instance.input_objects, set(action.args), set(failed_stream.instance.input_objects) & set(action.args))
            if 'calibrate' in action.name or 'take_image' in action.name:
                if len(set(failed_stream.instance.input_objects) & set(action.args)) == 2:
                    infeasible_action = action
                    break
        if infeasible_action is None:
            raise NotImplementedError(f'SESAME error: cannot identify the infeasible action for {failed_stream}')
        infeasible_action_name = infeasible_action.name
        pddl_plan = pddl_from_obj_plan(action_plan.action_plan)
        plan_str = '\n'.join(str(action) for action in pddl_plan)
        failed_stream_pddl_str = f'{failed_stream.external.name}:{pddl_list_from_expression(failed_stream.input_objects)}->{pddl_list_from_expression(failed_stream.external.outputs)}'
        infeasible_action_str = f'{infeasible_action_name}{pddl_list_from_expression(infeasible_action.args)}'

        error_str_backtracking = (
            "The system attempted the following plan in the past, but it failed:"
            f"\n\n```\n{plan_str}\n```\n\n"
            f"This was because the following function call did not succeed: `{failed_stream_pddl_str}`, "
            f"which impeded executing the action {infeasible_action_str}.\n"
        )
        return {
            # 'new_predicate': new_predicate,
            'new_preconditions': new_preconditions,
            'new_effects': new_effects,
            # 'new_init_atom': new_init_atom
            'remove_init_atom': remove_init_atom,
            'error_str_backtracking': error_str_backtracking,
        }
    
    else:
        # JMM: Means we didn't find a plan, so we will need to sample a new skeleton
        # JMM: goal will be to find the objects that cause failure and the action that
        # fails to be refined, and add a precondition to the action that those objects are "moved"
        print(f"Failed to sample stream at {highest_idx} / {len(stream_plan)} for {stream_plan[highest_idx]}")
        failed_stream = stream_plan[highest_idx]
        # JMM: Could be a failed test (the input objects are the ones that cause failure) or a filed sampler (the output objects are the ones that cause failure)
        failed_sample = [inp for inp in failed_stream.instance.input_objects if inp in free_objects] + \
                            [out for out in failed_stream.output_objects if out in free_objects]
        if len(failed_sample) == 0:
            new_preconditions = {}
            new_effects = {}
            remove_init_atom = None
            
            pddl_plan = pddl_from_obj_plan(action_plan.action_plan)
            plan_str = '\n'.join(str(action) for action in pddl_plan)
            failed_stream_pddl_str = f'{failed_stream.external.name}:{pddl_list_from_expression(failed_stream.input_objects)}->{pddl_list_from_expression(failed_stream.external.outputs)}'
            error_str_backtracking = (
                            "The system attempted the following plan in the past, but it failed:"
                            f"\n\n```\n{plan_str}\n```\n\n"
                            f"This was because the following function call did not succeed: `{failed_stream_pddl_str}`\n"
                        )
        else:
        
            # JMM: A hack -- check if value is int because in the PR2 domain, those are the movable objects
            infeasible_actions = [a for p in failed_sample for a in action_plan.action_plan if p in a.args ]
            # assert len(infeasible_actions) == 1, f'SESAME error: {len(infeasible_actions)} != 1.\n\t{infeasible_actions}'
            if len(infeasible_actions) == 1:
                infeasible_action = infeasible_actions[0]
            else:
                # JMM: Only one was actually infeasible. Maybe check additional parameters of the stream for matches with the action?
                # I'm assuming that the action that has most params in common with the failed stream is the one that caused the failure
                # This is just a guess and I didn't think this through
                infeasible_action = infeasible_actions[0]   # I guess this is the first one in the plan. This makes sense for at least grasps (where picks come before places)
                # infeasible_action = None
                # max_params_in_common = 1
                # for a in infeasible_actions:
                #     params_in_common = [p for p in a.args if p in failed_stream.instance.input_objects]
                #     if len(params_in_common) > max_params_in_common:
                #         infeasible_action = a
                #         max_params_in_common = len(params_in_common)
                # if infeasible_action is None:
                #     raise NotImplementedError(f'SESAME error: cannot identify the infeasible action')
            offended_objects = [p for p in infeasible_action.args if p not in free_objects and isinstance(p.value, int)]
            culprit_objects = [p for p in failed_stream.instance.input_objects if p not in free_objects and isinstance(p.value, int) and not p in offended_objects]
            offended_objects_idx = [idx for idx, p in enumerate(infeasible_action.args) if p not in free_objects and isinstance(p.value, int)]
            print("culprit objects:", culprit_objects)
            print("infeasible action:", infeasible_action)
            print("offended objects:", offended_objects)
            # print("domain ppdl:", domain.pddl)

            # JMM: Try this -- create a new predicate CausesFailure<action_name>, with parameters
            # culprit_objects and [a.args for a in infeasible_actions]. We will then add a precondition
            # to the action that forall x not CausesFailure<action_name>(x, a.args). We will also add
            # an effect to all actions that forall x not CausesFailure<action_name>(a.args, x). The 
            # missing bit will be to inform the function that called this that it should add a fluent 
            # to the initial state that CausesFailure<action_name>(culprit_objects, a.args)

            # Create the new predicate
            if len(culprit_objects) == 1:
                culprit_args = '?culprit'
            else:
                culprit_args = ' '.join([f'?culprit{i+1}' for i in range(len(culprit_objects))])
            infeasible_action_name = infeasible_action.name
            if len(offended_objects) == 1:
                offended_args = '?offended'
            else:
                offended_args = ' '.join([f'?offended{i+1}' for i in range(len(offended_objects))])

            # new_predicate = f'(CausesFailure{infeasible_action_name} {culprit_args} {offended_args})'
            # new_predicate = f'(CausesFailure{infeasible_action_name} {culprit_args})'
            # new_predicate = f'(NotCausesFailure{infeasible_action_name})'

            # Add the precondition to the action
            # Find the pddl action that corresponds to the infeasible action and pick the parameters whose indices are in offended_objects_idx
            pddl_action = [a for a in domain.actions if a.name == infeasible_action_name][0]
            offended_params = ' '.join([pddl_action.parameters[i].name for i in offended_objects_idx])
            if len(culprit_objects) > 0:
                # new_precondition = f'(forall ({culprit_args}) (not (CausesFailure{infeasible_action_name} {culprit_args} {offended_params})))'
                # new_precondition = f'(forall ({culprit_args}) (or (= {culprit_args} ?o) (not (CausesFailure{infeasible_action_name} {culprit_args}))))'
                # new_precondition = f'(forall ({culprit_args}) (or (or (= {culprit_args} ?o) (not (Graspable {culprit_args}))) (NotCausesFailure {culprit_args})))'
                new_precondition = f'(forall ({culprit_args}) (or (or (= {culprit_args} ?o) (not (Graspable {culprit_args}))) (NotCausesFailure {culprit_args} ?o)))'
            else:
                new_precondition = f'(NotCausesFailureSolo ?o)'
            new_preconditions = {infeasible_action_name: new_precondition}

            # Add the effect to any action that contains an effect that has the same parameters as the culprit_args
            # JMM: in theory, it could be possible that an effect has _more_ parameters and still affects the right things, but I'll ignore this
            # It's also possible that an effect with fewer parameters modifies one of them, but I'll also ignore this -- none of this is likely
            # to matter for these problems (or maybe any realistic problem). More realistically, we'd have types and do things properly based on
            # those
            new_effects = {}
            for a in domain.actions:
                if a.name == 'pick':
                    if len(culprit_objects) > 0:
                        # new_effects[a.name] = [f'(forall {offended_args} (not (CausesFailure{infeasible_action_name} ?o {offended_args})))']
                        # new_effects[a.name] = [f'(NotCausesFailure ?o)']
                        new_effects[a.name] = [f'(forall ({offended_args}) (when (Graspable {offended_args}) (NotCausesFailure ?o {offended_args})))']
                    else:
                        new_effects[a.name] = [f'(NotCausesFailureSolo ?o)']
                # for e in a.effects:
                #     if len(e.literal.args) == len(culprit_objects):
                #         culprit_params = ' '.join(e.literal.args)
                #         new_effect = f'(forall ({offended_args}) (not (CausesFailure{infeasible_action_name} {culprit_params} {offended_args})))'
                #         if a.name in new_effects:
                #             new_effects[a.name].append(new_effect)
                #         else:
                #             new_effects[a.name] = [new_effect]

            # Add the initial state
            # new_init_atom = (f'CausesFailure{infeasible_action_name}', *culprit_objects, *offended_objects)
            # new_init_atom = (f'CausesFailure{infeasible_action_name}', *culprit_objects)
            if len(culprit_objects) > 0:
                remove_init_atom = ('NotCausesFailure', *culprit_objects, *offended_objects)
            else:
                remove_init_atom = ('NotCausesFailureSolo', *offended_objects)
        
            pddl_plan = pddl_from_obj_plan(action_plan.action_plan)
            plan_str = '\n'.join(str(action) for action in pddl_plan)
            failed_stream_pddl_str = f'{failed_stream.external.name}:{pddl_list_from_expression(failed_stream.input_objects)}->{pddl_list_from_expression(failed_stream.external.outputs)}'
            infeasible_action_str = f'{infeasible_action_name}{pddl_list_from_expression(infeasible_action.args)}'

            error_str_backtracking = (
                "The system attempted the following plan in the past, but it failed:"
                f"\n\n```\n{plan_str}\n```\n\n"
                f"This was because the following function call did not succeed: `{failed_stream_pddl_str}`, "
                f"which impeded executing the action {infeasible_action_str}.\n"
            )
        return {
            # 'new_predicate': new_predicate,
            'new_preconditions': new_preconditions,
            'new_effects': new_effects,
            # 'new_init_atom': new_init_atom
            'remove_init_atom': remove_init_atom,
            'error_str_backtracking': error_str_backtracking,
        }


def process_stream_plan(store, domain, disabled, stream_plan, action_plan, cost,
                        bind=True, max_failures=0):
    # Bad old implementation of this method
    # The only advantage of this vs skeleton is that this can avoid the combinatorial growth in bindings
    if not is_plan(stream_plan):
        return
    if not stream_plan:
        store.add_plan(action_plan, cost)
        return
    stream_plan = [result for result in stream_plan if result.optimistic]
    free_objects = get_free_objects(stream_plan)
    bindings = {}
    bound_plan = []
    num_wild = 0
    for idx, opt_result in enumerate(stream_plan):
        if (store.best_cost <= cost) or (max_failures < (idx - len(bound_plan))):
            # TODO: this terminates early when bind=False
            break
        opt_inputs = [inp for inp in opt_result.instance.input_objects if inp in free_objects]
        if (not bind and opt_inputs) or not all(inp in bindings for inp in opt_inputs):
            continue
        bound_result = opt_result.remap_inputs(bindings)
        bound_instance = bound_result.instance
        if bound_instance.enumerated or not is_instance_ready(store.evaluations, bound_instance):
            continue
        # TODO: could remove disabled and just use complexity_limit
        new_results, new_facts = process_instance(store, domain, bound_instance) # TODO: bound_result
        num_wild += len(new_facts)
        if not bound_instance.enumerated:
            disabled.add(bound_instance)
        for new_result in new_results:
            if new_result.is_successful():
                bound_plan.append(new_results[0])
                bindings = update_bindings(bindings, bound_result, bound_plan[-1])
                cost = update_cost(cost, opt_result, bound_plan[-1])
                break
    if (num_wild == 0) and (len(stream_plan) == len(bound_plan)):
        store.add_plan(bind_action_plan(action_plan, bindings), cost)
    # TODO: report back whether to try w/o optimistic values in the event that wild

##################################################

# def process_stream_plan_branch(store, domain, disabled, stream_plan, action_plan, cost):
#     if not is_plan(stream_plan):
#         return
#     stream_plan = [result for result in stream_plan if result.optimistic]
#     if not stream_plan:
#         store.add_plan(action_plan, cost)
#         return
#     free_objects = get_free_objects(stream_plan)
#     bindings = defaultdict(set)
#     for opt_result in stream_plan:
#         opt_inputs = [inp for inp in opt_result.instance.input_objects if inp in free_objects]
#         inp_bindings = [bindings[inp] for inp in opt_inputs]
#         for combo in product(*inp_bindings):
#             bound_result = opt_result.remap_inputs(get_mapping(opt_inputs, combo))
#             bound_instance = bound_result.instance
#             if bound_instance.enumerated or not is_instance_ready(store.evaluations, bound_instance):
#                 continue # Disabled
#             new_results = process_instance(store, domain, bound_instance)
#             if not bound_instance.enumerated:
#                 disabled.add(bound_instance)
#             if isinstance(opt_result, StreamResult):
#                 for new_result in new_results:
#                     for out, obj in safe_zip(opt_result.output_objects, new_result.output_objects):
#                         bindings[out].add(obj)
#     #Binding = namedtuple('Binding', ['index', 'mapping'])
#     # TODO: after querying, search over all bindings of the produced sampled