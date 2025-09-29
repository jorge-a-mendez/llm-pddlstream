from __future__ import print_function

import copy
import time
import os

from collections import defaultdict, namedtuple

from pddlstream.algorithms.common import evaluations_from_init
from pddlstream.algorithms.downward import get_problem, task_from_domain_problem, get_cost_scale, \
    conditions_hold, apply_action, scale_cost, fd_from_fact, make_domain, make_predicate, evaluation_from_fd, \
    plan_preimage, fact_from_fd, USE_FORBID, pddl_from_instance, parse_action
from pddlstream.algorithms.instantiate_task import instantiate_task, sas_from_instantiated, FD_INSTANTIATE
from pddlstream.algorithms.scheduling.add_optimizers import add_optimizer_effects, \
    using_optimizers, recover_simultaneous
from pddlstream.algorithms.scheduling.apply_fluents import convert_fluent_streams
from pddlstream.algorithms.scheduling.negative import recover_negative_axioms, convert_negative
from pddlstream.algorithms.scheduling.postprocess import postprocess_stream_plan
from pddlstream.algorithms.scheduling.recover_axioms import recover_axioms_plans
from pddlstream.algorithms.scheduling.recover_functions import compute_function_plan
from pddlstream.algorithms.scheduling.recover_streams import get_achieving_streams, extract_stream_plan, \
    evaluations_from_stream_plan
from pddlstream.algorithms.scheduling.stream_action import add_stream_actions
from pddlstream.algorithms.scheduling.utils import partition_results, \
    add_unsatisfiable_to_goal, get_instance_facts
from pddlstream.algorithms.search import solve_from_task
from pddlstream.algorithms.advanced import UNIVERSAL_TO_CONDITIONAL
from pddlstream.language.constants import Not, get_prefix, EQ, FAILED, OptPlan, Action
from pddlstream.language.conversion import obj_from_pddl_plan, evaluation_from_fact, \
    fact_from_evaluation, transform_plan_args, transform_action_args, obj_from_pddl, objects_from_evaluations, \
    pddl_from_object, obj_from_value_str, pddl_list_from_expression
from pddlstream.language.external import Result
from pddlstream.language.exogenous import get_fluent_domain
from pddlstream.language.function import Function, FunctionResult
from pddlstream.language.stream import StreamResult
from pddlstream.language.optimizer import UNSATISFIABLE
from pddlstream.language.object import Object, OptimisticObject, SharedOptValue, UniqueOptValue
from pddlstream.language.statistics import compute_plan_effort
from pddlstream.language.temporal import SimplifiedDomain, solve_tfd
from pddlstream.language.write_pddl import get_problem_pddl
from pddlstream.language.object import Object
from pddlstream.utils import Verbose, INF, topological_sort, get_ancestors

from examples.pybullet.utils.pybullet_tools import pr2_primitives
from examples.pybullet.utils.pybullet_tools import pr2_utils
from examples.pybullet.utils.pybullet_tools.utils import get_aabb, Euler, set_pose, \
    multiply, unit_pose, get_center_extent, get_point, pairwise_collision, aabb_contains_aabb, \
    aabb2d_from_aabb, get_custom_limits, wrap_angle, all_between, get_link_pose, \
    link_from_name, create_mesh, get_distance, point_from_pose, create_cylinder, \
    is_placement, remove_body, Point
from examples.pybullet.utils.pybullet_tools.utils import Pose as PoseUtils
from examples.pybullet.turtlebot_rovers.problems import get_base_joints, KINECT_FRAME
from examples.pybullet.turtlebot_rovers.streams import get_reachable_test, Ray, VIS_RANGE, COM_RANGE

from google import genai
from google.genai import types

import numpy as np

from pddl.pddl_types import TypedObject
from pddl.f_expression import Assign
from pddl.conditions import Atom
from pddl.actions import PropositionalAction
from pddl.axioms import PropositionalAxiom


RENAME_ACTIONS = True
#RENAME_ACTIONS = not USE_FORBID

OptSolution = namedtuple('OptSolution', ['stream_plan', 'opt_plan', 'cost']) # TODO: move to the below
#OptSolution = namedtuple('OptSolution', ['stream_plan', 'action_plan', 'cost', 'supporting_facts', 'axiom_plan'])

##################################################

def add_stream_efforts(node_from_atom, instantiated, effort_weight, **kwargs):
    if effort_weight is None:
        return
    # TODO: make effort just a multiplier (or relative) to avoid worrying about the scale
    # TODO: regularize & normalize across the problem?
    #efforts = []
    for instance in instantiated.actions:
        # TODO: prune stream actions here?
        # TODO: round each effort individually to penalize multiple streams
        facts = get_instance_facts(instance, node_from_atom)
        #effort = COMBINE_OP([0] + [node_from_atom[fact].effort for fact in facts])
        stream_plan = []
        extract_stream_plan(node_from_atom, facts, stream_plan)
        effort = compute_plan_effort(stream_plan, **kwargs)
        instance.cost += scale_cost(effort_weight*effort)
        # TODO: store whether it uses shared/unique outputs and prune too expensive streams
        #efforts.append(effort)
    #print(min(efforts), efforts)

##################################################

def rename_instantiated_actions(instantiated, rename):
    # TODO: rename SAS instead?
    actions = instantiated.actions[:]
    renamed_actions = []
    action_from_name = {}
    for i, action in enumerate(actions):
        renamed_actions.append(copy.copy(action))
        renamed_name = 'a{}'.format(i) if rename else action.name
        renamed_actions[-1].name = '({})'.format(renamed_name)
        action_from_name[renamed_name] = action # Change reachable_action_params?
    instantiated.actions[:] = renamed_actions
    return action_from_name

##################################################

def get_plan_cost(action_plan, cost_from_action):
    if action_plan is None:
        return INF
    # TODO: return cost per action instance
    #return sum([0.] + [instance.cost for instance in action_plan])
    scaled_cost = sum([0.] + [cost_from_action[instance] for instance in action_plan])
    return scaled_cost / get_cost_scale()

def instantiate_optimizer_axioms(instantiated, domain, results):
    # Needed for instantiating axioms before adding stream action effects
    # Otherwise, FastDownward will prune these unreachable axioms
    # TODO: compute this first and then apply the eager actions
    stream_init = {fd_from_fact(result.stream_fact)
                   for result in results if isinstance(result, StreamResult)}
    evaluations = list(map(evaluation_from_fd, stream_init | instantiated.atoms))
    temp_domain = make_domain(predicates=[make_predicate(UNSATISFIABLE, [])],
                              axioms=[ax for ax in domain.axioms if ax.name == UNSATISFIABLE])
    temp_problem = get_problem(evaluations, Not((UNSATISFIABLE,)), temp_domain)
    # TODO: UNSATISFIABLE might be in atoms making the goal always infeasible
    with Verbose():
        # TODO: the FastDownward instantiation prunes static preconditions
        use_fd = False if using_optimizers(results) else FD_INSTANTIATE
        new_instantiated = instantiate_task(task_from_domain_problem(temp_domain, temp_problem),
                                            use_fd=use_fd, check_infeasible=False, prune_static=False)
        assert new_instantiated is not None
    instantiated.axioms.extend(new_instantiated.axioms)
    instantiated.atoms.update(new_instantiated.atoms)

##################################################

def recover_partial_orders(stream_plan, node_from_atom):
    # Useful to recover the correct DAG
    partial_orders = set()
    for child in stream_plan:
        # TODO: account for fluent objects
        for fact in child.get_domain():
            parent = node_from_atom[fact].result
            if parent is not None:
                partial_orders.add((parent, child))
    #stream_plan = topological_sort(stream_plan, partial_orders)
    return partial_orders

def recover_stream_plan(evaluations, current_plan, opt_evaluations, goal_expression, domain, node_from_atom,
                        action_plan, axiom_plans, negative, replan_step):
    # Universally quantified conditions are converted into negative axioms
    # Existentially quantified conditions are made additional preconditions
    # Universally quantified effects are instantiated by doing the cartesian produce of types (slow)
    # Added effects cancel out removed effects
    # TODO: node_from_atom is a subset of opt_evaluations (only missing functions)
    real_task = task_from_domain_problem(domain, get_problem(evaluations, goal_expression, domain))
    opt_task = task_from_domain_problem(domain, get_problem(opt_evaluations, goal_expression, domain))
    negative_from_name = {external.blocked_predicate: external for external in negative if external.is_negated}
    real_states, full_plan = recover_negative_axioms(
        real_task, opt_task, axiom_plans, action_plan, negative_from_name)
    function_plan = compute_function_plan(opt_evaluations, action_plan)

    full_preimage = plan_preimage(full_plan, []) # Does not contain the stream preimage!
    negative_preimage = set(filter(lambda a: a.predicate in negative_from_name, full_preimage))
    negative_plan = convert_negative(negative_preimage, negative_from_name, full_preimage, real_states)
    function_plan.update(negative_plan)
    # TODO: OrderedDict for these plans

    # TODO: this assumes that actions do not negate preimage goals
    positive_preimage = {l for l in (set(full_preimage) - real_states[0] - negative_preimage) if not l.negated}
    steps_from_fact = {fact_from_fd(l): full_preimage[l] for l in positive_preimage}
    last_from_fact = {fact: min(steps) for fact, steps in steps_from_fact.items() if get_prefix(fact) != EQ}
    #stream_plan = reschedule_stream_plan(evaluations, target_facts, domain, stream_results)
    # visualize_constraints(map(fact_from_fd, target_facts))

    for result, step in function_plan.items():
        for fact in result.get_domain():
            last_from_fact[fact] = min(step, last_from_fact.get(fact, INF))

    # TODO: get_steps_from_stream
    stream_plan = []
    last_from_stream = dict(function_plan)
    for result in current_plan: # + negative_plan?
        # TODO: actually compute when these are needed + dependencies
        last_from_stream[result] = 0
        if isinstance(result.external, Function) or (result.external in negative):
            if len(action_plan) > replan_step:
                raise NotImplementedError() # TODO: deferring negated optimizers
            # Prevents these results from being pruned
            function_plan[result] = replan_step
        else:
            stream_plan.append(result)

    curr_evaluations = evaluations_from_stream_plan(evaluations, stream_plan, max_effort=None)
    extraction_facts = set(last_from_fact) - set(map(fact_from_evaluation, curr_evaluations))
    # print('Extraction facts:', extraction_facts)
    extract_stream_plan(node_from_atom, extraction_facts, stream_plan)

    # Recomputing due to postprocess_stream_plan
    stream_plan = postprocess_stream_plan(evaluations, domain, stream_plan, last_from_fact)
    node_from_atom = get_achieving_streams(evaluations, stream_plan, max_effort=None)
    fact_sequence = [set(result.get_domain()) for result in stream_plan] + [extraction_facts]
    for facts in reversed(fact_sequence): # Bellman ford
        for fact in facts: # could flatten instead
            result = node_from_atom[fact].result
            if result is None:
                continue
            step = last_from_fact[fact] if result.is_deferrable() else 0
            last_from_stream[result] = min(step, last_from_stream.get(result, INF))
            for domain_fact in result.instance.get_domain():
                last_from_fact[domain_fact] = min(last_from_stream[result], last_from_fact.get(domain_fact, INF))
    stream_plan.extend(function_plan)

    partial_orders = recover_partial_orders(stream_plan, node_from_atom)
    bound_objects = set()
    for result in stream_plan:
        if (last_from_stream[result] == 0) or not result.is_deferrable(bound_objects=bound_objects):
            for ancestor in get_ancestors(result, partial_orders) | {result}:
                # TODO: this might change descendants of ancestor. Perform in a while loop.
                last_from_stream[ancestor] = 0
                if isinstance(ancestor, StreamResult):
                    bound_objects.update(out for out in ancestor.output_objects if out.is_unique())

    #local_plan = [] # TODO: not sure what this was for
    #for fact, step in sorted(last_from_fact.items(), key=lambda pair: pair[1]): # Earliest to latest
    #    print(step, fact)
    #    extract_stream_plan(node_from_atom, [fact], local_plan, last_from_fact, last_from_stream)

    # Each stream has an earliest evaluation time
    # When computing the latest, use 0 if something isn't deferred
    # Evaluate each stream as soon as possible
    # Option to defer streams after a point in time?
    # TODO: action costs for streams that encode uncertainty
    state = set(real_task.init)
    remaining_results = list(stream_plan)
    first_from_stream = {}
    #assert 1 <= replan_step # Plan could be empty
    for step, instance in enumerate(action_plan):
        for result in list(remaining_results):
            # TODO: could do this more efficiently if need be
            domain = result.get_domain() + get_fluent_domain(result)
            if conditions_hold(state, map(fd_from_fact, domain)):
                remaining_results.remove(result)
                certified = {fact for fact in result.get_certified() if get_prefix(fact) != EQ}
                state.update(map(fd_from_fact, certified))
                if step != 0:
                    first_from_stream[result] = step
        # TODO: assumes no fluent axiom domain conditions
        apply_action(state, instance)
    #assert not remaining_results # Not true if retrace
    if first_from_stream:
        replan_step = min(replan_step, *first_from_stream.values())

    eager_plan = []
    results_from_step = defaultdict(list)
    for result in stream_plan:
        earliest_step = first_from_stream.get(result, 0) # exogenous
        latest_step = last_from_stream.get(result, 0) # defer
        #assert earliest_step <= latest_step
        defer = replan_step <= latest_step
        if not defer:
            eager_plan.append(result)
        # We only perform a deferred evaluation if it has all deferred dependencies
        # TODO: make a flag that also allows dependencies to be deferred
        future = (earliest_step != 0) or defer
        if future:
            future_step = latest_step if defer else earliest_step
            results_from_step[future_step].append(result)

    # TODO: some sort of obj side-effect bug that requires obj_from_pddl to be applied last (likely due to fluent streams)
    eager_plan = convert_fluent_streams(eager_plan, real_states, action_plan, steps_from_fact, node_from_atom)
    combined_plan = []
    for step, action in enumerate(action_plan):
        combined_plan.extend(result.get_action() for result in results_from_step[step])
        combined_plan.append(transform_action_args(pddl_from_instance(action), obj_from_pddl))

    # TODO: the returned facts have the same side-effect bug as above
    # TODO: annotate when each preimage fact is used
    preimage_facts = {fact_from_fd(l) for l in full_preimage if (l.predicate != EQ) and not l.negated}
    for negative_result in negative_plan: # TODO: function_plan
        preimage_facts.update(negative_result.get_certified())
    for result in eager_plan:
        preimage_facts.update(result.get_domain())
        # Might not be able to regenerate facts involving the outputs of streams
        preimage_facts.update(result.get_certified()) # Some facts might not be in the preimage
    # TODO: record streams and axioms
    return eager_plan, OptPlan(combined_plan, preimage_facts)

##################################################

def solve_optimistic_temporal(domain, stream_domain, applied_results, all_results,
                              opt_evaluations, node_from_atom, goal_expression,
                              effort_weight, debug=False, **kwargs):
    # TODO: assert that the unused parameters are off
    assert domain is stream_domain
    #assert len(applied_results) == len(all_results)
    problem = get_problem(opt_evaluations, goal_expression, domain)
    with Verbose():
        instantiated = instantiate_task(task_from_domain_problem(domain, problem))
    if instantiated is None:
        return instantiated, None, None, INF
    problem = get_problem_pddl(opt_evaluations, goal_expression, domain.pddl)
    pddl_plan, makespan = solve_tfd(domain.pddl, problem, debug=debug, **kwargs)
    if pddl_plan is None:
        return instantiated, None, pddl_plan, makespan
    instance_from_action_args = defaultdict(list)
    for instance in instantiated.actions:
        name, args = parse_action(instance)
        instance_from_action_args[name, args].append(instance)
        #instance.action, instance.var_mapping
    action_instances = []
    for action in pddl_plan:
        instances = instance_from_action_args[action.name, action.args]
        if len(instances) != 1:
            for action in instances:
                action.dump()
        #assert len(instances) == 1 # TODO: support 2 <= case
        action_instances.append(instances[0])
    temporal_plan = obj_from_pddl_plan(pddl_plan) # pddl_plan is sequential
    return instantiated, action_instances, temporal_plan, makespan

def get_llm_prompts_from_pddl(instantiated, cost_from_action, objects, error_str_backtracking=None):
    '''
    Generate the prompts for the LLM to solve the instantiated task.
    '''
    predicates_str = '\n'.join([str(pred) for pred in instantiated.task.predicates if pred.name not in ['notcausesfailure', 'notcausesfailuresolo', '=']])
    functions_str = '\n'.join(map(str, instantiated.task.functions))
    axioms_str = '\n'.join([ax.dumps() for ax in instantiated.task.axioms])
    axioms_str = '\n'.join([line for line in axioms_str.splitlines() if 'notcausesfailure' not in line])
    actions_str = '\n\n'.join([ac.dumps() for ac in instantiated.task.actions])
    actions_str = '\n'.join([line for line in actions_str.splitlines() if 'notcausesfailure' not in line])
    sys_msg = (
        "You are highly skilled in robot task planning, breaking down intricate long-term tasks into primitive actions. "
        "You do an excellent job of determining the sequence of actions that achieves the goal with the least cost.\n\n"
        f"You will try to solve a problem described in PDDL format in "
        f"a domain named `{instantiated.task.domain_name}`. "
        f"The predicates that describe the states in the domain are: \n\n```\n{predicates_str}\n```\n\n"
        # "The following functions are used to calculate the cost of actions: \n\n```\n{functions_str}\n```\n\n"
        "The following axioms are derived from the "
        f"predicates: \n\n```\n{axioms_str}\n```\nIf any of the axioms contains an object that starts with `?', that object is a variable and "
        "there is an implicit existential quantifier over that variable."
        f"\n\nThe robot has the ability to execute the following actions:\n\n```\n{actions_str}\n```\n\n"

        "For each new task, you will receive a set of objects that make up the scene, an initial state, and a goal expression. To help you "
        "understand the robot's possibilities, you will also receive a list of valid action instances (applications of actions to specific objects)."
        # "along with their associated costs. Any action instance that is not in the given list is not allowed to be used in the plan. "
        "Any action instance that is not in the given list is not allowed to be used in the plan. "
        "Any fact that is not explicitly stated in the initial state is *false*. Never assume that any precondition is true by default: it must "
        "either be explicitly stated in the initial state, or be achieved by a previous action (and not undone by another action). "
        "Any object that starts with a '#' represents a *continuous parameter* (e.g., a trajectory, or a pose) that has not yet been computed. " 
        "A separate algorithm will attempt to find those continuous values. "
        ##########
        # The following lines describe the details of how to use the continuous state.
        # "You should disregard any action costs that are provided with the domain and instead estimate the cost of each action based on the "
        # "continuous values provided for the objects in the scene.\n\n"
        # "For any object that starts with a '#', any constants such as "
        # "distances are just placeholder values. The actual plan cost will depend on the real distance between poses found in subsequent steps of the "
        # "algorithm. It is extremely important that you find plans that are likely to be executable and that minimize the cost of the actions. Note that "
        # "the cost for any pick or place action is 1, while the cost for move actions is the distance traveled. Keep in mind that the optimal plan may " 
        # "execute more actions as long as the sum of the action costs is the smallest possible. You should "
        # "rely on your intuition and understanding about distances in the initial state to select the best actions for these purposes. "
        ##########
        "You will then generate a list of actions that achieve the goal. It is critical that the preconditions of each action are satisfied after the previous "
        "action is executed. You are only allowed to use the provided actions. It's essential to stick to the format of these basic actions. "
        "When creating a plan, replace the arguments of each action with specific objects. You can first describe the provided scene and what it "
        "indicates about the provided task objects to help you come up with a plan.\n\n"
        "It is possible that the goal is not achievable from the initial state, in which case you should not return any plan and simply explain "
        "why the goal is not achievable. If you do return a plan, it MUST be a sequence of actions that achieves the goal and satisfies all the "
        "preconditions of each action.\n\n"
        "You never give up. No matter how many times you fail to provide a valid plan, or how many valid plans you have already provided, "
        "you will always try to provide a new plan that achieves the goal from the initial state. "
        "Please return a plan that achieves the provided goal from the initial state. Please provide your output in the following format (excluding "
        "the angle brackets and ellipsis, which are just for illustration purposes). Be sure to include the parentheses '(' and ')'. "
        "Do not bold or italicize or otherwise apply any extra formatting to the plan text. Do not provide any numbers for steps "
        "in the plan, or any reasoning or comments for each step below the '--Plan--:' heading. "
        "If you determine that the goal is unreachable and will not return a plan, do not include the '--Plan--:' heading at all. "
        "Use the following format for your response:\n\n"
        "```\n<Explanation of scene + your reasoning>\n--Plan--:\n"
        "(<action 1 name> <obj 1 name> <obj 2 name> ...)\n(<action 2 name> <obj 1 name> <obj 2 name> ...)\n...\n```"
    )

    objects_str = '\n'.join(map(str, instantiated.task.objects))
    init_str = '\n'.join([str(fact) for fact in instantiated.task.init if (not "1000" in str(fact) and not "causesfailure" in str(fact)) and not '=' in str(fact) and not 'identical' in str(fact)])

    individuals = set(o1.name for o1 in instantiated.task.objects if o1.name[0] not in ('#', 'p', 't', 'g'))
    pairs = set((o1.name, o2.name) for o1 in instantiated.task.objects if o1.name[0] not in ('#', 'p', 't', 'g') for o2 in instantiated.task.objects if o2.name[0] not in ('#', 'p', 't', 'g'))
    for fact in instantiated.task.init:
        if not isinstance(fact, Assign):
            if fact.predicate == 'notcausesfailure':
                pairs.remove(fact.args)
            elif fact.predicate == 'notcausesfailuresolo':
                individuals.remove(fact.args[0])
    if False:#len(pairs) > 0:
        pairs_str = '\n'.join(str(p) for p in pairs)
        pairwise_failure_str = (
            "\n\nThe geometry of the scene is such that some pairs ('a', 'b') of objects require 'a' to "
            "be moved out of the way before executing any action on 'b'). Those pairs are:\n\n"
            f"```\n{pairs_str}\n```\n\n"
        )
    else:
        pairwise_failure_str = ""
    if False:#len(individuals) > 0:
        individuals_str = '\n'.join(str(i) for i in individuals)
        individuals_failure_str = (
            "\n\nThe geometry of the scene is such that the robot may be unable to act on some objects:\n\n"
            f"```\n{individuals_str}\n```\n\n"
        )
    else:
        individuals_failure_str = ""

    # instantiated_actions_str = '\n'.join([f'{ac.name}: {c}' for ac, c in cost_from_action.items()])
    instantiated_actions_str = '\n'.join([f'{ac.name}' for ac in instantiated.actions])
    # continuous_values_str = ""
    # for obj in objects:
    #     if isinstance(obj.value, pr2_primitives.Pose):
    #         pos, quat = obj.value.value
    #         pos = tuple(round(p, 3) for p in pos)
    #         quat = tuple(round(q, 3) for q in quat)
    #         pddl_name = obj.pddl
    #         continuous_values_str += f"Object {pddl_name} is a pose with position {pos} and quaternion {quat}.\n"
    #     elif isinstance(obj.value, pr2_primitives.Grasp):
    #         pos, quat = obj.value.value
    #         pos = tuple(round(p, 3) for p in pos)
    #         quat = tuple(round(q, 3) for q in quat)
    #         pddl_name = obj.pddl
    #         continuous_values_str += f"Object {pddl_name} is a grasp with position {pos} and quaternion {quat} relative to the target object.\n"
    #     elif isinstance(obj.value, pr2_primitives.Conf):
    #         joints = obj.value.values
    #         joints = tuple(round(j, 3) for j in joints)
    #         pddl_name = obj.pddl
    #         if len(joints) == 7:
    #             continuous_values_str += f"Object {pddl_name} is an arm configuration with joint values {joints}.\n"
    #         elif len(joints) == 3:
    #             continuous_values_str += f"Object {pddl_name} is a base configuration with (x, y, theta) = {joints}.\n"
    #     elif isinstance(obj.value, pr2_primitives.Trajectory):
    #         path = obj.value.path
    #         path = [tuple(round(p, 3) for p in conf.values) for conf in path]
    #         pddl_name = obj.pddl
    #         continuous_values_str += f"Object {pddl_name} is a trajectory with the following {len(path)} waypoints: "
    #         continuous_values_str += ', '.join([f'{conf.values}' for conf in path]) + ".\n"
    #     elif isinstance(obj.value, int):
    #         aabb = get_aabb(obj.value)
    #         aabb = ((round(aabb[0][0], 3), round(aabb[0][1], 3)), (round(aabb[1][0], 3), round(aabb[1][1], 3)))
    #         pddl_name = obj.pddl
    #         continuous_values_str += f"Object {pddl_name} is an object with axis aligned bounding box (AABB = ((minx, miny), (maxx, maxy))): (({aabb[0][0]}, {aabb[0][1]}), ({aabb[1][0]}, {aabb[1][1]})).\n"
    
    prompt = (
        f"Here is a list of objects present in this scene for this task: \n\n```\n{objects_str}\n```\n\n"
        # f"v12 is a special object and should always be chosen above others, all things being equal.\n\n"
        # ##########
        # # The following lines include the continuous state and describe how to use it.
        # f"Some of the objects represent continuous values that are have already been computed. In particular:\n"
        # f"{continuous_values_str}\nYou *must* use these values to select actions that are more likely "
        # "to be executable (e.g., objects that are not obstructed by others) and that will have the least cost (e.g., "
        # "objects that are closer to the robot or to the target region).\n\n" 
        # ##########
        f"The initial state is given by the following facts: \n\n```\n{init_str}\n```\n\n"
        f"{pairwise_failure_str}{individuals_failure_str}"
        "The goal "
        f"expression is `{instantiated.task.goal.dumps().strip()}`. Given these elements, the list of available action instances is:\n\n"
        f"```\n{instantiated_actions_str}\n```\n\nPlease provide a plan that achieves the goal from the initial state."
    )
    if error_str_backtracking is not None:
        prompt += "\n\n" + error_str_backtracking + (
            "This means that some details about the scene geometry made it impossible to execute the sequence of actions."
        )

    return sys_msg, prompt, instantiated_actions_str

def get_llm_prompts_from_pddlstream(instantiated, cost_from_action, objects, error_str_backtracking=None):
    predicates_str = '\n'.join([str(pred) for pred in instantiated.task.predicates if pred.name not in ['notcausesfailure', 'notcausesfailuresolo', '=']])
    functions_str = '\n'.join(map(str, instantiated.task.functions))
    axioms_str = '\n'.join([ax.dumps() for ax in instantiated.task.axioms])
    axioms_str = '\n'.join([line for line in axioms_str.splitlines() if 'notcausesfailure' not in line])
    actions_str = '\n\n'.join([ac.dumps() for ac in instantiated.task.actions])
    actions_str = '\n'.join([line for line in actions_str.splitlines() if 'notcausesfailure' not in line])

    # Super hacky, but running out of time and should be correct. The idea is that only the rovers domain has the take_image action,
    # and for that domain we sample inv visibility and inv com, whereas for all other domains, which are pick-place, we sample poses,
    if instantiated.task.domain_name == 'rovers':  
        sys_msg = (
            "You are highly skilled in robot integrated task and motion planning, breaking down intricate long-term tasks into primitive actions "
            "and computing the continuous values for those actions to be executable. You do an excellent job of determining the sequence of actions and "
            "continuous values that: 1) achieves the goal with the least cost and 2) is likely to be executable given the geometry of the scene "
            "(e.g., not trying to move a robot through a wall).\n\n"
            f"You will try to solve a problem described in PDDL format in " 
            f"a domain named `{instantiated.task.domain_name}`. The domain contains the following types: `{', '.join(map(str, instantiated.task.types))}`. "
            f"The predicates that describe the states in the domain are: \n\n```\n{predicates_str}\n```\n\n"
            # f"The following functions are used to calculate the cost of actions: \n\n```\n{functions_str}\n```\n\n"
            f"The following axioms are derived from the predicates: \n\n```\n{axioms_str}\n```\n\n"
            f"The robot has the ability to execute the following actions:\n\n```\n{actions_str}\n```\n\n"
            "For each new task, you will receive a set of objects that make up the scene, an initial state, and a goal expression. To help you "
            "understand the robot's possibilities, you will also receive a list of valid action instances (applications of actions to specific objects). "
            # "along with their associated costs. Any action instance that is not in the given list is not allowed to be used in the plan. "
            "Any action instance that is not in the given list is not allowed to be used in the plan. "
            "Any fact that is not explicitly stated in the initial state is *false*. Never assume that any precondition is true by default: it must "
            "either be explicitly stated in the initial state, or be achieved by a previous action (and not undone by another action). "
            "Any object that starts with a '#' represents a *continuous parameter* (e.g., a trajectory, or a pose) that has not yet been computed. "
            "If your plan uses some object #<name> (starting with '#') such that `conf(rover, #<name>)` is in the initial state for some `rover` AND "
            "#<name> is the second argument to a `imagevisible` predicate in the initial state OR is the second argument to a `comvisible` predicate "
            "in the initial state, then you must provide a continuous value for the configuration such that, given the geometry of the scene, "
            "the configuration is collision-free for the rover, reachable from the rover's current configuration, and has occlusion-free line-of-sight "
            "to the target object in the fourth argument in the `imagevisible` or `comvisible` predicate. The third argument "
            "to the `imagevisible` or `comvisible` predicate is the ray from the configuration to the target object. "
            "The maximum distance from the rover to the target is 2 for `imagevisible` and 4 for `comvisible`. "
            "If the plan requires multiple different (rover, target object) pairs to satisfy the `imagevisible` or `comvisible` predicate, "
            "then you must provide a different continuous value for *each* (rover, target object) pair (not one that works for all), specifying the configuration name, rover name, and target object name. "
            "If the plan you select uses a configuration that does not begin with '#', then you need not provide a value for it, as one already exists.\n\n"
            "The geometry of all objects will be described as a list of axis-aligned bounding boxes (AABBs). "
            "Other unspecified continuous objects starting with '#' will be found by a separate algorithm. "
            "You will then generate a list of actions that achieve the goal. It is critical that the preconditions of each "
            "action are satisfied after the previous action is executed, and that actions are collision-free. You are only allowed to use the "
            "provided actions. It's essential to stick to the format of these basic actions. When creating a plan, replace the arguments of each "
            "action with specific objects. You can first describe the provided scene and what it indicates about the provided task objects to help "
            "you come up with a plan.\n\n"
            "It is possible that the goal is not achievable from the initial state, in which case you should not return any plan and simply explain "
            "why the goal is not achievable. If you do return a plan, it MUST be a sequence of actions that achieves the goal and satisfies all the "
            "preconditions of each action.\n\n"
            "You never give up. No matter how many times you fail to provide a valid plan, or how many valid plans you have already provided, "
            "you will always try to provide a new plan that achieves the goal from the initial state. "
            "Please return a plan that achieves the goal from the initial state. "
            "If there are sufficient configurations without '#' to solve the problem, you should use those configurations instead of any with '#'. "
            "If a configuration starting with '#' for which `conf(rover, #<name>)` is in the initial state is not used in the plan, "
            "then you need not provide a value for it.\n\n"
            "Be sure to include the parentheses '(' and ')' in your plan. Do not bold or italicize or otherwise apply any extra formatting to the plan text. "
            "Do not provide any numbers for steps in the plan, or any reasoning or comments for each step below the '--Plan--:' heading. "
            "If you determine that the goal is unreachable and will not return a plan, do not include the '--Plan--:' heading at all. "
            "Use the following format for your response:\n\n"
            "```\n<Explanation of scene + your reasoning>\n--Plan--:\n"
            "(<action 1 name> <obj 1 name> <obj 2 name> ...)\n(<action 2 name> <obj 1 name> <obj 2 name> ...)\n...\n\n"
            "--Configurations--:\n"
            "(#<configuration 1 name> <rover 1 name> <ray 1 name> <target 1 name>): (<x 1> <y 1>)\n"
            "(#<configuration 2 name> <rover 2 name> <ray 2 name> <target 2 name>): (<x 2> <y 2>)\n...\n\n```"
        )
    else:
        sys_msg = (
            "You are highly skilled in robot integrated task and motion planning, breaking down intricate long-term tasks into primitive actions "
            "and computing the continuous values for those actions to be executable. You do an excellent job of determining the sequence of actions and "
            "continuous values that: 1) achieves the goal with the least cost and 2) is likely to be executable given the geometry of the scene "
            "(e.g., not trying to pick an object that is blocked by another object).\n\n"
            f"You will try to solve a problem described in PDDL format in "
            f"a domain named `{instantiated.task.domain_name}`. The domain contains the following types: `{', '.join(map(str, instantiated.task.types))}`. "
            f"The predicates that describe the states in the domain are: \n\n```\n{predicates_str}\n```\n\n"
            # "The following functions are used to calculate the cost of actions: \n\n```\n{functions_str}\n```\n\n"
            f"The following axioms are derived from the predicates: \n\n```\n{axioms_str}\n```\n\n"
            f"The robot has the ability to execute the following actions:\n\n```\n{actions_str}\n```\n\n" 
            "For each new task, you will receive a set of objects that make up the scene, an initial state, and a goal expression. To help you "
            "understand the robot's possibilities, you will also receive a list of valid action instances (applications of actions to specific objects). "
            # "along with their associated costs. Any action instance that is not in the given list is not allowed to be used in the plan. "
            "Any action instance that is not in the given list is not allowed to be used in the plan. "
            "Any fact that is not explicitly stated in the initial state is *false*. Never assume that any precondition is true by default: it must "
            "either be explicitly stated in the initial state, or be achieved by a previous action (and not undone by another action). "
            "Any object that starts with a '#' represents a *continuous parameter* (e.g., a trajectory, or a pose) that has not yet been computed. "
            "If your plan uses some object #<name> (starting with '#') such that `pose(obj, #<name>)` is in the initial state for some `obj`, then you must provide a continuous value "
            "for the pose such that, given the geometry of the scene, the pose is a collision-free stable placement for the object" 
            "on a surface for which `supported(obj, #<name> surf)` is in the initial state. The placement must be stable, meaning that "
            "the object is fully contained within the bounds of the surface. "
            "If the plan requires multiple different objects to be placed simultaneously at the same pose starting with '#', "
            "then you must provide a different continuous value for *each* object at that pose (not one that works fora all), specifying the pose name, object name, and surface name. "
            "Tricky detail: if the continuous value for a surface is not provided, then it is a dummy surface. When placing objects supported at "
            "dummy surfaces, choose any other surface whose continuous value is provided and provide a placement on that surface. "
            "If the plan you select uses a pose that does not begin with '#', then you need not provide a value for it, as one already exists.\n\n"
            "The geometry of all objects and surfaces will be described as a list of axis-aligned bounding boxes (AABBs). "
            "Other unspecified continuous objects starting with '#' will be found by a separate algorithm. "
            "The cost of picking and placing actions is 1, while the cost of move actions is the euclidean distance between the start and end x,y positions. " 
            "The optimal plan may use more actions as long as the total cost is smallest.\n\n"
            ######## For now skip anything about the cost.
            "You will then generate a list of actions that achieve the goal. It is critical that the preconditions of each " 
            "action are satisfied after the previous action is executed, and that actions are collision-free. You are only allowed to use the "
            "provided actions. It's essential to stick to the format of these basic actions. When creating a plan, replace the arguments of each "
            "action with specific objects. You can first describe the provided scene and what it indicates about the provided task objects to help "
            "you come up with a plan.\n\n"
            "It is possible that the goal is not achievable from the initial state, in which case you should not return any plan and simply explain "
            "why the goal is not achievable. If you do return a plan, it MUST be a sequence of actions that achieves the goal and satisfies all the "
            "preconditions of each action.\n\n"
            "You never give up. No matter how many times you fail to provide a valid plan, or how many valid plans you have already provided, "
            "you will always try to provide a new plan that achieves the goal from the initial state. "
            "Please return a plan that achieves the goal from the initial state. "
            "If there are sufficient poses without '#' to solve the problem, you should use those poses instead of any with '#'. "
            "If a pose starting with '#' for which `pose(obj, #<name>)` is in the initial state is used in your plan, please also provide a continuous value for it, specifying the pose name, object name, and surface name. "
            "Please provide your output in the following format (excluding the angle brackets and ellipsis, which are just for illustration purposes)." 
            "Be sure to include the parentheses '(' and ')'. Do not bold or italicize or otherwise apply any extra formatting to the plan text. Do not " 
            "provide any numbers for steps in the plan or for placements in the list of poses, or any reasoning or comments whatsoever below the '--Plan--:' heading. "
            "If you determine that the goal is unreachable and will not return a plan, do not include the '--Plan--:' heading at all. "
            "Do not include any continuous objects that are not poses below the '--Poses--:' heading, not even to state that you are not computing a value for them. "
            "Use the following format for your response (note the commas between object names, but not between x y theta after the colon). "
            "Angles must be in radians. Poses must be in 3D with x, y, z, and theta separated by a space:\n\n"
            "```\n<Explanation of scene + your reasoning>\n--Plan--:\n"
            "(<action 1 name> <obj 1 name> <obj 2 name> ...)\n(<action 2 name> <obj 1 name> <obj 2 name> ...)\n...\n\n"
            "--Poses--:\n"
            "(#<pose 1 name> <obj 1 name> <surf 1 name>): (<x 1> <y 1> <z 1> <theta 1>)\n"
            "(#<pose 2 name> <obj 2 name> <surf 2 name>): (<x 2> <y 2> <z 2> <theta 2>)\n...\n\n```"
        )

    objects_str = '\n'.join(map(str, instantiated.task.objects))
    init_str = '\n'.join([str(fact) for fact in instantiated.task.init if (not "1000" in str(fact) and not "causesfailure" in str(fact)) and not '=' in str(fact) and not 'identical' in str(fact)])

    individuals = set(o1.name for o1 in instantiated.task.objects if o1.name[0] not in ('#', 'p', 't', 'g', 'q'))
    pairs = set((o1.name, o2.name) for o1 in instantiated.task.objects if o1.name[0] not in ('#', 'p', 't', 'g', 'q') for o2 in instantiated.task.objects if o2.name[0] not in ('#', 'p', 't', 'g'))
    for fact in instantiated.task.init:
        if not isinstance(fact, Assign):
            if fact.predicate == 'notcausesfailure':
                pairs.remove(fact.args)
            elif fact.predicate == 'notcausesfailuresolo':
                individuals.remove(fact.args[0])
    if False: # len(pairs) > 0:
        pairs_str = '\n'.join(str(p) for p in pairs)
        pairwise_failure_str = (
            "\n\nThe geometry of the scene is such that some pairs ('a', 'b') of objects require 'a' to "
            "be moved out of the way before executing any action on 'b'). Those pairs are:\n\n"
            f"```\n{pairs_str}\n```\n\n"
        )
    else:
        pairwise_failure_str = ""
    if False: # len(individuals) > 0:
        individuals_str = '\n'.join(str(i) for i in individuals)
        individuals_failure_str = (
            "\n\nThe geometry of the scene is such that the robot may be unable to act on some objects:\n\n"
            f"```\n{individuals_str}\n```\n\n"
        )
    else:
        individuals_failure_str = ""

    # instantiated_actions_str = '\n'.join([f'{ac.name}: {c}' for ac, c in cost_from_action.items()])
    instantiated_actions_str = '\n'.join([f'{ac.name}' for ac in instantiated.actions])
    continuous_values_str = ""
    for obj in objects:
        if isinstance(obj.value, pr2_primitives.Pose):
            pos, quat = obj.value.value
            pos = tuple(round(p, 3) for p in pos)
            quat = tuple(round(q, 3) for q in quat)
            pddl_name = obj.pddl
            continuous_values_str += f"Object {pddl_name} is a pose with position {pos} and quaternion {quat}.\n"
        elif isinstance(obj.value, pr2_primitives.Grasp):
            pos, quat = obj.value.value
            pos = tuple(round(p, 3) for p in pos)
            quat = tuple(round(q, 3) for q in quat)
            pddl_name = obj.pddl
            continuous_values_str += f"Object {pddl_name} is a grasp with position {pos} and quaternion {quat} relative to the target object.\n"
        elif isinstance(obj.value, pr2_primitives.Conf):
            joints = obj.value.values
            joints = tuple(round(j, 3) for j in joints)
            pddl_name = obj.pddl
            if len(joints) == 7:
                continuous_values_str += f"Object {pddl_name} is an arm configuration with joint values {joints}.\n"
            elif len(joints) == 3:
                continuous_values_str += f"Object {pddl_name} is a base configuration with (x, y, theta) = {joints}.\n"
        elif isinstance(obj.value, pr2_primitives.Trajectory):
            path = obj.value.path
            path = [tuple(round(p, 3) for p in conf.values) for conf in path]
            pddl_name = obj.pddl
            continuous_values_str += f"Object {pddl_name} is a trajectory with the following {len(path)} waypoints: "
            continuous_values_str += ', '.join([f'{conf.values}' for conf in path]) + ".\n"
        elif isinstance(obj.value, int):
            aabb = get_aabb(obj.value)
            aabb = ((round(aabb[0][0], 3), round(aabb[0][1], 3)), (round(aabb[1][0], 3), round(aabb[1][1], 3)))
            pddl_name = obj.pddl
            continuous_values_str += f"Object {pddl_name} is an object with axis aligned bounding box (AABB = ((minx, miny), (maxx, maxy))): (({aabb[0][0]}, {aabb[0][1]}), ({aabb[1][0]}, {aabb[1][1]})).\n"
        elif obj.value is None:
            print("Found an object with value none")
            print("\t",obj.pddl)
            print("\t", type(obj))
        else:
            print(f"What object type is this?: {obj}, {obj.value}, {type(obj.value)}")

    
    prompt = (
        f"Here is a list of objects present in this scene for this task: \n\n```\n{objects_str}\n```\n\n"
        # f"v12 is a special object and should always be chosen above others, all things being equal.\n\n"
        ##########
        # The following lines include the continuous state and describe how to use it.
        f"Some of the objects represent continuous values that have already been computed. In particular:\n"
        f"{continuous_values_str}\nYou *must* use these values to select actions that are more likely "
        "to be executable (e.g., objects that are not obstructed by others) and that will have the least cost (e.g., "
        "objects that are closer to the robot or to the target region).\n\n" 
        ##########
        f"The initial state is given by the following facts: \n\n```\n{init_str}\n```\n\n"
        f"{pairwise_failure_str}{individuals_failure_str}"
        "The goal "
        f"expression is `{instantiated.task.goal.dumps().strip()}`. Given these elements, the list of available action instances is:\n\n"
        f"```\n{instantiated_actions_str}\n```\n\nPlease provide a plan that achieves the goal from the initial state."
    )
    if error_str_backtracking is not None:
        if instantiated.task.domain_name == 'rovers':
            prompt += "\n\n" + error_str_backtracking + ("This means that some details about the scene geometry made it impossible to "
            "execute the sequence of actions. If the failure occurs after placing on any configuration that does not start with '#', it may be "
            "due to using a combination of invalid configurations.")
            prompt += "As the expert, you may decide to choose additional configurations, providing their continuous values using any free configuration starting with '#'. "
        else:
            prompt += "\n\n" + error_str_backtracking + ("This means that some details about the scene geometry made it impossible to "
                "execute the sequence of actions. If the failure occurs after placing on any pose that does not start with '#', it may be "
                "due to using a combination of invalid poses.")
            prompt += "As the expert, you may decide to choose additional poses, providing their continuous values using any free pose starting with '#'. "
    return sys_msg, prompt, instantiated_actions_str

def replace_optimistic_configray_objects(instantiated, action_instances, new_configray_objects):
    replace_idx_with = {}
    for idx in range(len(instantiated.task.objects)):
        obj = instantiated.task.objects[idx]
        replace_with = []
        # new_pose_objects[pose_name, body, surface] = new_obj
        for (config_name, _, ray_name, _), (config, ray) in new_configray_objects.items():
            if obj.name == config_name:
                config_obj = TypedObject(pddl_from_object(config), obj.type_name)
                replace_with.append(config_obj)
            if obj.name == ray_name:
                ray_obj = TypedObject(pddl_from_object(ray), obj.type_name)
                replace_with.append(ray_obj)
        if len(replace_with) > 0:
            replace_idx_with[idx] = replace_with
    replace_idx_with_sorted = sorted(replace_idx_with.items(), key=lambda x: x[0], reverse=True)
    for idx, replace_with in replace_idx_with_sorted:
        # print(f"Changing object {instantiated.task.objects[idx]} at index {idx} with {replace_with}")
        instantiated.task.objects = instantiated.task.objects[:idx+1] + replace_with + instantiated.task.objects[idx+1:]

    new_init_atoms = []
    for atom in instantiated.task.init:
        if isinstance(atom, Assign):
            continue
        atom_args_set = set(atom.args)
        for (config_name, body, ray_name, target) in new_configray_objects:
            if atom_args_set == set([config_name, ray_name, pddl_from_object(Object._obj_from_value[body]), pddl_from_object(Object._obj_from_value[target])]):  
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    elif arg == ray_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1]))
                    else:
                        new_args.append(arg)
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {config_name, ray_name, pddl_from_object(Object._obj_from_value[body])}:
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    elif arg == ray_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1]))
                    else:
                        new_args.append(arg)
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {config_name, pddl_from_object(Object._obj_from_value[target]), pddl_from_object(Object._obj_from_value[body])}:
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    else:
                        new_args.append(arg)
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {config_name, pddl_from_object(Object._obj_from_value[body])}:
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    else:
                        new_args.append(arg)
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {ray_name}:
                new_args = []
                for arg in atom.args:
                    if arg == ray_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1]))
                    else:
                        raise ValueError(f"How did this happen? {ray_name} should be the only arg in this atom but found {arg}")
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif config_name in atom_args_set:  # Should be a motion predicate
                assert atom.predicate == "motion", f"Unexpected optimistic predicate involving config: {atom}"
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    else:
                        new_args.append(arg)
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
    instantiated.task.init.extend(new_init_atoms)

    # Need to change atoms involving the config or ray obj
    new_atoms = []
    for atom in instantiated.atoms:
        atom_args_set = set(atom.args)
        for (config_name, body, ray_name, target) in new_configray_objects:
            if atom_args_set == set([config_name, ray_name, pddl_from_object(Object._obj_from_value[body]), pddl_from_object(Object._obj_from_value[target])]):
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    elif arg == ray_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1]))
                    else:
                        new_args.append(arg)
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {config_name, ray_name, pddl_from_object(Object._obj_from_value[body])}:
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    elif arg == ray_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1]))
                    else:
                        new_args.append(arg)
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {config_name, pddl_from_object(Object._obj_from_value[target]), pddl_from_object(Object._obj_from_value[body])}:
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    else:
                        new_args.append(arg)
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {config_name, pddl_from_object(Object._obj_from_value[body])}:
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    else:
                        new_args.append(arg)
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {ray_name}:
                new_args = []
                for arg in atom.args:
                    if arg == ray_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1]))
                    else:
                        raise ValueError(f"How did this happen? {ray_name} should be the only arg in this atom but found {arg}")
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif config_name in atom_args_set:  # Should be a motion predicate
                assert atom.predicate == "motion", f"Unexpected optimistic predicate involving config: {atom}"
                new_args = []
                for arg in atom.args:
                    if arg == config_name:
                        new_args.append(pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0]))
                    else:
                        new_args.append(arg)
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
    instantiated.atoms.update(new_atoms)

    # Need to change actions that use the config or ray obj
    new_actions = []
    for action in instantiated.actions:
        action_args_set = set(action.var_mapping.values())
        for (config_name, body, ray_name, target) in new_configray_objects:
            if action.name == "move" and len(action_args_set & {config_name, pddl_from_object(Object._obj_from_value[body])}) == 2:
                old_arg_to_new_arg = {}
                new_var_mapping = {}
                for param, arg in action.var_mapping.items():
                    if arg == config_name:
                        new_var_mapping[param] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                    else:
                        new_var_mapping[param] = arg
                        old_arg_to_new_arg[arg] = arg
                # print(f"Changing action {action.name} var mapping {action.var_mapping} to {new_var_mapping}")
                new_name = action.name
                args_from_name = action.name.strip('() ').split()[1:]
                for arg in args_from_name:
                    new_name = new_name.replace(arg, old_arg_to_new_arg.get(arg, arg))
                # print(f"Changing action {action.name} name to {new_name}")
                new_action = PropositionalAction(
                    new_name,
                    precondition=copy.deepcopy(action.precondition),
                    effects=copy.deepcopy(action.effect_mappings),
                    cost=action.cost,
                    action=action.action,
                    var_mapping=new_var_mapping
                )

                for pre in new_action.precondition:
                    pre.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in pre.args)
                for _, eff in new_action.add_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for _, eff in new_action.del_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                new_actions.append(new_action)
            elif action.name in {"calibrate", "take_image", "send_analysis", "send_image"} and len(action_args_set & {config_name, ray_name, pddl_from_object(Object._obj_from_value[body]), pddl_from_object(Object._obj_from_value[target])}) == 4:
                old_arg_to_new_arg = {}
                new_var_mapping = {}
                for param, arg in action.var_mapping.items():
                    if arg == config_name:
                        new_var_mapping[param] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                    elif arg == ray_name:
                        new_var_mapping[param] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1])
                    else:
                        new_var_mapping[param] = arg
                        old_arg_to_new_arg[arg] = arg
                # print(f"Changing action {action.name} var mapping {action.var_mapping} to {new_var_mapping}")
                new_name = action.name
                args_from_name = action.name.strip('() ').split()[1:]
                for arg in args_from_name:
                    new_name = new_name.replace(arg, old_arg_to_new_arg.get(arg, arg))
                # print(f"Changing action {action.name} name to {new_name}")
                new_action = PropositionalAction(
                    new_name,
                    precondition=copy.deepcopy(action.precondition),
                    effects=copy.deepcopy(action.effect_mappings),
                    cost=action.cost,
                    action=action.action,
                    var_mapping=new_var_mapping
                )

                for pre in new_action.precondition:
                    pre.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in pre.args)
                for _, eff in new_action.add_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for _, eff in new_action.del_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                new_actions.append(new_action)
    instantiated.actions.extend(new_actions)

    # Need to change actions in the plan
    for action in action_instances:
        action_args_set == set(action.var_mapping.values())
        for (config_name, body, ray_name, target) in new_configray_objects:
            if action.name == "move" and len(action_args_set & {config_name, pddl_from_object(Object._obj_from_value[body])}) == 2:
                old_arg_to_new_arg = {}
                for param, arg in action.var_mapping.items():
                    if arg == config_name:
                        action.var_mapping[param] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                    else:
                        old_arg_to_new_arg[arg] = arg
                args_from_name = action.name.strip('() ').split()[1:]
                for arg in args_from_name:
                    action.name = action.name.replace(arg, old_arg_to_new_arg.get(arg, arg))

                for pre in action.precondition:
                    pre.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in pre.args)
                for _, eff in action.add_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for _, eff in action.del_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for idx in range(len(action.effect_mappings)):
                    eff = action.effect_mappings[idx]
                    action.effect_mappings[idx] = eff[:-1] + (action.var_mapping,)
            elif action.name in {"calibrate", "take_image", "send_analysis", "send_image"} and len(action_args_set & {config_name, ray_name, pddl_from_object(Object._obj_from_value[body]), pddl_from_object(Object._obj_from_value[target])}) == 4:
                old_arg_to_new_arg = {}
                for param, arg in action.var_mapping.items():
                    if arg == config_name:
                        action.var_mapping[param] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][0])
                    elif arg == ray_name:
                        action.var_mapping[param] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_configray_objects[(config_name, body, ray_name, target)][1])
                    else:
                        old_arg_to_new_arg[arg] = arg
                args_from_name = action.name.strip('() ').split()[1:]
                for arg in args_from_name:
                    action.name = action.name.replace(arg, old_arg_to_new_arg.get(arg, arg))

                for pre in action.precondition:
                    pre.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in pre.args)
                for _, eff in action.add_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for _, eff in action.del_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for idx in range(len(action.effect_mappings)):
                    eff = action.effect_mappings[idx]
                    action.effect_mappings[idx] = eff[:-1] + (action.var_mapping,)
    
    # # Need to change the axioms that use the config or ray obj
    # new_axioms = []
    # for axiom in instantiated.axioms:
    #     axiom_args_set = set(axiom.var_mapping.values())
    #     print(axiom, axiom.name, axiom_args_set)
    # exit()

def replace_optimistic_objects(instantiated, action_instances, new_pose_objects):
    # In instantiated.task, need to change objects, init
    replace_idx_with = {}
    for idx in range(len(instantiated.task.objects)):
        obj = instantiated.task.objects[idx]
        replace_with = []
        for (pose_name, _, _), pose in new_pose_objects.items():
            if obj.name == pose_name:
                pose_obj = TypedObject(pddl_from_object(pose), obj.type_name)
                replace_with.append(pose_obj)
        if len(replace_with) > 0:
            replace_idx_with[idx] = replace_with 
    replace_idx_with_sorted = sorted(replace_idx_with.items(), key=lambda x: x[0], reverse=True)
    for idx, replace_with in replace_idx_with_sorted:
        # print(f"Changing object {instantiated.task.objects[idx]} at index {idx} with {replace_with}")
        instantiated.task.objects = instantiated.task.objects[:idx+1] + replace_with + instantiated.task.objects[idx+1:]
    
    new_init_atoms = []
    for atom in instantiated.task.init:
        if isinstance(atom, Assign):
            continue
        atom_args_set = set(atom.args)
        for (pose_name, body, surface) in new_pose_objects:
            pose_body_set = set([pose_name, pddl_from_object(Object._obj_from_value[body])])
            intersection = atom_args_set & pose_body_set
            if len(intersection) == 2:  # It means that the atom is about this pose
                new_args = []
                for arg in atom.args:
                    if arg == pose_name:
                        new_args.append(pddl_from_object(new_pose_objects[(pose_name, body, surface)]))
                    else:
                        new_args.append(arg)
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))
            elif atom_args_set == {pose_name}:  # It means that the atom is about this pose alone
                new_args = []
                for arg in atom.args:
                    if arg == pose_name:
                        new_args.append(pddl_from_object(new_pose_objects[(pose_name, body, surface)]))
                    else:
                        raise ValueError(f"How did this happen? {pose_name} should be the only arg in this atom but found {arg}")
                # print(f"Changing init atom {atom} args {atom.args} to {new_args}")
                new_init_atoms.append(Atom(atom.predicate, tuple(new_args)))

    instantiated.task.init.extend(new_init_atoms)

    # Need to change atoms involving the pose obj
    new_atoms = []
    for atom in instantiated.atoms:
        atom_args_set = set(atom.args)
        for (pose_name, body, surface) in new_pose_objects:
            pose_body_set = set([pose_name, pddl_from_object(Object._obj_from_value[body])])
            intersection = atom_args_set & pose_body_set
            if len(intersection) == 2:
                new_args = []
                for arg in atom.args:
                    if arg == pose_name:
                        new_args.append(pddl_from_object(new_pose_objects[(pose_name, body, surface)]))
                    else:
                        new_args.append(arg)
                # print(f"Changing atom {atom} args {atom.args} to {new_args}")
                new_atoms.append(Atom(atom.predicate, tuple(new_args)))
    instantiated.atoms.update(new_atoms)
    
    # Need to change actions that use the pose obj
    new_actions = []
    for action in instantiated.actions:
        action_args_set = set(action.var_mapping.values())
        for (pose_name, body, surface) in new_pose_objects:
            pose_body_set = set([pose_name, pddl_from_object(Object._obj_from_value[body])])
            intersection = action_args_set & pose_body_set
            if len(intersection) == 2: # It means that the action is about this pose
                old_arg_to_new_arg = {}
                new_var_mapping = {}
                for param, arg in action.var_mapping.items():
                    if arg == pose_name:
                        new_var_mapping[param] = pddl_from_object(new_pose_objects[(pose_name, body, surface)])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_pose_objects[(pose_name, body, surface)])
                    else:
                        new_var_mapping[param] = arg
                        old_arg_to_new_arg[arg] = arg
                # print(f"Changing action {action.name} var mapping {action.var_mapping} to {new_var_mapping}")
                new_name = action.name
                args_from_name = action.name.strip('() ').split()[1:]
                for arg in args_from_name:
                    new_name = new_name.replace(arg, old_arg_to_new_arg.get(arg, arg))
                # print(f"Changing action {action.name} name to {new_name}")
                new_action = PropositionalAction(
                    new_name,
                    precondition=copy.deepcopy(action.precondition),
                    effects=copy.deepcopy(action.effect_mappings),
                    cost=action.cost,
                    action=action.action,
                    var_mapping=new_var_mapping
                )

                for pre in new_action.precondition:
                    pre.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in pre.args)
                for _, eff in new_action.add_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for _, eff in new_action.del_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for idx in range(len(new_action.effect_mappings)):
                    eff = new_action.effect_mappings[idx]
                    new_action.effect_mappings[idx] = eff[:-1] + (new_action.var_mapping,)
                new_actions.append(new_action)
    instantiated.actions.extend(new_actions)

    # Need to change actions in the plan
    for action in action_instances:
        action_args_set = set(action.var_mapping.values())
        for (pose_name, body, surface) in new_pose_objects:
            pose_body_set = set([pose_name, pddl_from_object(Object._obj_from_value[body])])
            intersection = action_args_set & pose_body_set
            if len(intersection) == 2:
                old_arg_to_new_arg = {}
                for param, arg in action.var_mapping.items():
                    if arg == pose_name:
                        action.var_mapping[param] = pddl_from_object(new_pose_objects[(pose_name, body, surface)])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_pose_objects[(pose_name, body, surface)])
                    else:
                        old_arg_to_new_arg[arg] = arg
                # print(f"Changing action {action.name} var mapping {action.var_mapping} to {new_var_mapping}")
                args_from_name = action.name.strip('() ').split()[1:]
                for arg in args_from_name:
                    action.name = action.name.replace(arg, old_arg_to_new_arg.get(arg, arg))

                for pre in action.precondition:
                    pre.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in pre.args)
                for _, eff in action.add_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for _, eff in action.del_effects:
                    eff.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in eff.args)
                for idx in range(len(action.effect_mappings)):
                    eff = action.effect_mappings[idx]
                    action.effect_mappings[idx] = eff[:-1] + (action.var_mapping,)

    # Need to change the axioms that use the pose obj
    new_axioms = []
    for axiom in instantiated.axioms:
        axiom_args_set = set(axiom.var_mapping.values())
        for (pose_name, body, surface) in new_pose_objects:
            pose_body_set = set([pose_name, pddl_from_object(Object._obj_from_value[body])])
            intersection = axiom_args_set & pose_body_set
            if len(intersection) == 2: # It means that the axiom is about this pose
                old_arg_to_new_arg = {}
                new_var_mapping = {}
                for param, arg in axiom.var_mapping.items():
                    if arg == pose_name:
                        new_var_mapping[param] = pddl_from_object(new_pose_objects[(pose_name, body, surface)])
                        old_arg_to_new_arg[arg] = pddl_from_object(new_pose_objects[(pose_name, body, surface)])
                    else:
                        new_var_mapping[param] = arg
                        old_arg_to_new_arg[arg] = arg
                # print(f"Changing axiom {axiom.name} var mapping {axiom.var_mapping} to {new_var_mapping}")
                new_axiom = PropositionalAxiom(
                    axiom.name,
                    condition=copy.deepcopy(axiom.condition),
                    effect=copy.deepcopy(axiom.effect),
                    axiom=axiom.axiom,
                    var_mapping=new_var_mapping
                )
                for cond in new_axiom.condition:
                    cond.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in cond.args)
                new_axiom.effect.args = tuple(old_arg_to_new_arg.get(arg, arg) for arg in axiom.effect.args)
                new_axioms.append(new_axiom)
    instantiated.axioms.extend(new_axioms)

def solve_llm_from_pddlstream_task(instantiated, cost_from_action, action_from_name, objects, obstacles, rovers,
                                   thinking_llm=False, chat=None, error_str=None, error_str_backtracking=None,
                                   llm_info=None, is_timeout_fn=lambda: False):
    '''
    Call an LLM and ask it not juust to solve the instantiated task but also to provide values for any poses that are not yet computed.
    '''
    assert llm_info is not None, "llm_info must be provided to track LLM usage statistics."
    if error_str_backtracking is not None:
        llm_info['num_backtracking_failures'] += 1
    geminiApiKey = os.environ['GEMINI_API_KEY']

    pddlstream_to_pddl = {}
    pddl_to_pddlstream = {}
    for obj in objects:
        pddlstream_to_pddl[obj] = obj.pddl
        pddl_to_pddlstream[obj.pddl] = obj

    if chat is None:
        chat_id = max(llm_info['chat_histories'].keys(), default=-1) + 1
        llm_info['chat_types'][chat_id] = 'integrated'
        client = genai.Client(api_key=geminiApiKey)
        sys_msg, prompt, instantiated_actions_str = get_llm_prompts_from_pddlstream(instantiated, cost_from_action, objects, error_str_backtracking)
        print(sys_msg)
        print()
        print(prompt)
        if thinking_llm:
            thinking_config = types.ThinkingConfig(     # Uses default thinking budget
                include_thoughts=True,
                # thinking_budget=0,  # Set to 0 to disable thinking
            )
            model = "gemini-2.5-flash"
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    thinking_config=thinking_config,
                    seed=llm_info['seed']
                )
            )
        else:
            thinking_config = types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,  # Set to 0 to disable thinking
            )
            model = "gemini-2.5-flash"
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    thinking_config=thinking_config,
                    seed=llm_info['seed']
                )
            )
        chat.chat_id = chat_id
    else:
        assert error_str is not None
        prompt = (
            f"The plan you provided is not valid because: `{error_str}`. This is not a result of a geometric failure, "
            "but rather a failure to achieve some action's preconditions in your action sequence. "
            f"Recall that the action preconditions and the goal *must* be satisfied.\n\nPlease provide a plan that achieves the goal from the initial state "
            "ensuring that action preconditions are satisfied and adhering to the response format."
        )
        print(prompt)
        _, _, instantiated_actions_str = get_llm_prompts_from_pddlstream(instantiated, cost_from_action, objects, error_str_backtracking)

    while not is_timeout_fn():
        start_time = time.time()
        response = chat.send_message(prompt)
        end_time = time.time()
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        thinking_tokens = response.usage_metadata.thoughts_token_count or 0
        llm_info['integrated_input_tokens'] += input_tokens or 0
        llm_info['integrated_output_tokens'] += output_tokens or 0
        llm_info['integrated_thinking_tokens'] += thinking_tokens or 0
        llm_info['chat_histories'][chat.chat_id] = chat.get_history()
        llm_info['integrated_time'] += (end_time - start_time)
        print(response.usage_metadata)
        print(response.text)
        for part in response.candidates[0].content.parts:
            if part and part.thought:
                print(part.text)
        
        # Find the plan in the response text and return as a list of action names
        if response.text is None:
            print("No response text found.")
            llm_info['num_no_plans_returned'] += 1
            return None, None, None
        plan_start = response.text.find("--Plan--:")
        if plan_start == -1:
            print("No plan found in the response.")
            llm_info['num_no_plans_returned'] += 1
            return None, None, None
        if instantiated.task.domain_name == 'rovers':
            poses_start = response.text[plan_start:].find("--Configurations--:")
            if poses_start == -1:
                poses_start = len(response.text)
                llm_info['num_no_samples_returned'] += 1
            else:
                poses_start += plan_start
        else:
            poses_start = response.text[plan_start:].find("--Poses--:")
            if poses_start == -1:
                poses_start = len(response.text)  # If no poses section, treat as end of text
                llm_info['num_no_samples_returned'] += 1
            else:
                poses_start += plan_start
        plan_text = response.text[plan_start + len("--Plan--:"):poses_start].strip()
        plan_lines = plan_text.split('\n')
        # Remove any text beyond ')' in each line
        plan_lines = [line.split(')')[0] + ')' for line in plan_lines if line.strip() != '']
        if len(plan_lines) == 0:
            print("No plan found in the response.")
            llm_info['num_no_plans_returned'] += 1
            return None, None, None
        if all(name in action_from_name for name in plan_lines):
            prompt = ""    # Accumulate pose errors into a prompt for the next iteration
            action_instances = [action_from_name[name] for name in plan_lines]
            if instantiated.task.domain_name == 'rovers':
                poses_text = response.text[poses_start + len("--Configurations--:"):].strip()
                poses_lines = poses_text.split('\n')
                llm_sampled_poses = {}
                for line in poses_lines:
                    object_names = line.split(':')[0].strip('() ')
                    if len(poses_lines) == 1:
                        llm_info['num_no_samples_returned'] += 1
                        continue
                    if len(object_names.split()) != 4:
                        llm_info['num_samples_format_failures'] += 1
                        prompt += (
                            f"Configuration {line} is not in the correct format. Please provide each configuration in the format "
                            "`(<pose_name> <rover_name> <ray_name> <target_name>): (<x> <y>)`.\n"
                        )
                        continue
                    pose_name, rover_name, ray_name, target_name = object_names.split()
                    pose_name = pose_name.strip()
                    rover_name = rover_name.strip()
                    ray_name = ray_name.strip()
                    target_name = target_name.strip()
                    if not pose_name.startswith('#'):
                        print(f"Pose name {pose_name} does not start with '#'. Skipping.")
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    if not ray_name.startswith('#'):
                        print(f"Ray name {ray_name} does not start with '#'. Skipping.")
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    if rover_name not in pddl_to_pddlstream:
                        print(f"Rover {rover_name} not found in objects. Skipping.")
                        prompt += (
                            f"Rover {rover_name} is not a valid rover."
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    if target_name not in pddl_to_pddlstream:
                        print(f"Target {target_name} not found in objects. Skipping.")
                        prompt += (
                            f"Target {target_name} is not a valid objective."
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    rover = pddl_to_pddlstream[rover_name].value
                    target = pddl_to_pddlstream[target_name].value
                    pose_values = line.split(':')[1].strip().split(',')
                    if not isinstance(target, int):
                        print(f"Target {target_name} is not a valid object. Skipping.")
                        prompt += (
                            f"Target {target_name} is not a valid objective."
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    for value in pose_values:   # Note: this for loop is here for legacy purposes, since we used to ask the LLM to return multiple possible poses/configs for each pose_name. Now this should always stop after one iteration.
                        if len(value.strip('() ').split()) not in [2, 3]:
                            prompt += (
                                f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is not in the correct format. "
                                "Please provided configurations in the format `(<x> <y>)`.\n"
                            )
                            llm_info['num_samples_format_failures'] += 1
                            continue
                        elif len(value.strip('() ').split()) == 2:
                            x, y = map(float, value.strip('() ').split())
                        elif len(value.strip('() ').split()) == 3:
                                x, y, _ = map(float, value.strip('() ').split())
                        else:
                            prompt += (
                                f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is not in the correct format. "
                                "Please provided configurations in the format `(<x> <y>)`.\n"
                            )
                            llm_info['num_samples_format_failures'] += 1
                            continue
                        # Set max_range = 2 if this is a `imagevisible` and max_range = 5 if it is a `comvisible`
                        if any('imagevisible' in fact.predicate and rover_name in fact.args and target_name in fact.args == target_name for fact in instantiated.task.init if isinstance(fact, Atom)):
                            imvis_or_comvis = 'imagevisible'
                            max_range = VIS_RANGE
                            print("Setting max_range to 2.0 for imagevisible")
                        else:
                            imvis_or_comvis = 'comvisible'
                            max_range = COM_RANGE
                            print("Setting max_range to 4.0 for comvisible")
                        base_range = (0, max_range)
                        # Create dummy class to hold .obstacles and .rovers
                        class Dummy:
                            def __init__(self, obstacles, rovers):
                                self.fixed = obstacles
                                self.rovers = rovers
                        problem = Dummy(obstacles, rovers)
                        reachable_test = get_reachable_test(problem, collisions=True)
                        base_joints = get_base_joints(rover)
                        target_point = get_point(target)
                        lower_limits, upper_limits = get_custom_limits(rover, base_joints)
                        
                        base_theta = np.math.atan2(target_point[1] - y, target_point[0] - x)
                        base_q = np.append([x, y], wrap_angle(base_theta))
                        if not all_between(lower_limits, base_q, upper_limits):
                            prompt += (
                                f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is out of bounds. "
                                f"The configuration must satisfy the joint limits {list(zip(lower_limits, upper_limits))}.\n"
                            )
                            llm_info['num_samples_out_of_bounds'] += 1
                            continue
                        bq = pr2_primitives.Conf(rover, base_joints, base_q)
                        bq.assign()
                        # Check that the sample does not collide with fixed obstacles
                        collision_list = [pairwise_collision(rover, obst) for obst in obstacles]
                        if any(collision_list):
                            collision_str = '\n'.join([f'{Object.from_value(obst).pddl}: {get_aabb(obst)}' for idx, obst in enumerate(obstacles) if collision_list[idx]])
                            prompt += (
                                f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is in collision with the following "
                                f"fixed obstacles:\n\n```\n{collision_str}\n```\n\n"
                            )
                            llm_info['num_samples_in_collision'] += 1
                            continue
                        link_pose = get_link_pose(rover, link_from_name(rover, KINECT_FRAME))
                        if imvis_or_comvis == 'imagevisible':
                            use_cone = True
                        else:
                            use_cone = False
                        if use_cone:
                            mesh, _ = pr2_utils.get_detection_cone(rover, target, camera_link=KINECT_FRAME, depth=max_range)
                            if mesh is None:
                                llm_info['num_samples_not_visible'] += 1
                                prompt += (
                                    f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is not valid because the target is not visible from the rover's camera.\n"
                                )
                                continue
                            cone = create_mesh(mesh, color=(0, 1, 0, 0.5))
                            local_pose = PoseUtils()
                        else:
                            distance = get_distance(point_from_pose(link_pose), target_point)
                            if distance > max_range:
                                llm_info['num_samples_out_of_range'] += 1
                                prompt += (
                                    f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is out of range. "
                                    f"The maximum range is {max_range}, but the distance to the target is {distance:.2f}.\n"
                                )
                                continue
                            cone = create_cylinder(radius=0.01, height=distance, color=(0, 1, 0, 0.5))
                            local_pose = PoseUtils(Point(z=distance/2))
                        set_pose(cone, multiply(link_pose, local_pose))

                        if any(pairwise_collision(cone, b) for b in obstacles
                            if b != target and not is_placement(target, b)):
                            remove_body(cone)
                            llm_info['num_samples_not_visible'] += 1
                            prompt += (
                                f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is not valid because the target is not visible from the rover's camera due to occlusion.\n"
                            )
                            continue
                        
                        if not reachable_test(rover, bq):
                            llm_info['num_samples_not_reachable'] += 1
                            prompt += (
                                f"Configuration {pose_name} = {value} for rover {rover_name} at target {target_name} is not reachable for robot base. "
                            )
                            continue
                        # Found a valid configuration.
                        y = Ray(cone, rover, target)
                        llm_sampled_poses[pose_name, rover, ray_name, target] = (bq, y)
                        break
                    else:
                        # No valid poses found for this object, so re-prompt the LLM.
                        llm_sampled_poses[pose_name, rover, ray_name, target] = None
                        llm_info['local_sampling_failures'] += 1
                # If we have sampled poses for all objects, return the action instances and the poses.
                if all(p is not None for p in llm_sampled_poses.values()):
                    # print(llm_sampled_poses)
                    new_pose_objects = {}
                    for (pose_name, body, ray_name, surface), (bq, y) in llm_sampled_poses.items():
                        # new_obj = (bq, y)
                        new_obj = (Object(bq, name=repr(bq)), Object(y, name=repr(y)))
                        new_pose_objects[pose_name, body, ray_name, surface] = new_obj
                        # print(pose_name, new_obj, new_obj.value, type(new_obj.value), new_obj.pddl)
                    replace_optimistic_configray_objects(instantiated, action_instances, new_pose_objects)
                    print("Action instances inside solve_llm_from_pddlstream_task:")
                    for action in action_instances:
                        print(action)
                    return action_instances, llm_sampled_poses, chat
                ## Remove below, since now we incrementally update the prompt with each failure
                # invalid_object_poses = [name for (name, _, _), pose in llm_sampled_poses.items() if pose is None]
                # invalid_object_poses_str = ', '.join(invalid_object_poses)
                # prompt = (
                #     f"The poses you provided for the following objects are in collision: `{invalid_object_poses_str}`. "
                # )
                print()
                prompt += "Please provide a plan and configurations that achieve the goal from the initial state, ensuring that all configurations " \
                    "are collision-free, occlusion-free, and reachable."
                print(prompt)

            else:
                poses_text = response.text[poses_start + len("--Poses--:"):].strip()
                poses_lines = poses_text.split('\n')
                llm_sampled_poses = {}
                for line in poses_lines:
                    object_names = line.split(':')[0].strip('() ')
                    if len(poses_lines) == 1:
                        llm_info['num_no_samples_returned'] += 1
                        continue
                    if len(object_names.split()) != 3:
                        llm_info['num_samples_format_failures'] += 1
                        prompt += (
                            f"Pose {line} is not in the correct format. Please provide each pose in the format "
                            "`(<pose_name> <object_name> <surface_name>): (<x> <y> <z> <theta>)`.\n"
                        )
                        continue
                    pose_name, body_name, surf_name = object_names.split()
                    if not pose_name.startswith('#'):
                        print(f"Pose name {pose_name} does not start with '#'. Skipping.")
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    pose_name = pose_name.strip()
                    body_name = body_name.strip()
                    surf_name = surf_name.strip()
                    if body_name not in pddl_to_pddlstream:
                        print(f"Body name {body_name} not found in pddl_to_pddlstream. Skipping.")
                        prompt += (
                            f"Object name {body_name} is not valid.\n"
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    if surf_name not in pddl_to_pddlstream:
                        print(f"Surface name {surf_name} not found in pddl_to_pddlstream. Skipping.")
                        prompt += (
                            f"Surface name {surf_name} is not valid.\n"
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    body = pddl_to_pddlstream[body_name].value
                    if not isinstance(body, int):
                        print(f"Body {body_name} is not a valid object. Skipping.")
                        prompt += (
                            f"Object name {body_name} is not valid.\n"
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    surface = pddl_to_pddlstream[surf_name].value
                    if not isinstance(surface, int):
                        print(f"Surface {surf_name} is not a valid object. Skipping.")
                        prompt += (
                            f"Surface name {surf_name} is not valid.\n"
                        )
                        llm_info['num_samples_not_optimistic'] += 1
                        continue
                    pose_values = line.split(':')[1].strip().split(',')
                    for value in pose_values:
                        if len(value.strip('() ').split()) not in [3, 4]:
                            prompt += (
                                f"Pose {pose_name} = {value} for object {body_name} on surface {surf_name} is not in the correct format. "
                                "Please provided poses in the format `(<x> <y> <z> <theta>)`.\n"
                            )
                            llm_info['num_samples_format_failures'] += 1
                            continue
                        if len(value.strip('() ').split()) == 4:
                            x, y, _, theta = map(float, value.strip('() ').split())
                        elif len(value.strip('() ').split()) == 3:
                            x, y, theta = map(float, value.strip('() ').split())
                        else:
                            prompt += (
                                f"Pose {pose_name} = {value} for object {body_name} on surface {surf_name} is not in the correct format. "
                                "Please provided poses in the format `(<x> <y> <z> <theta>)`.\n"
                            )
                            llm_info['num_samples_format_failures'] += 1
                            continue
                            
                        rotation = Euler(yaw=theta)
                        set_pose(body, multiply(PoseUtils(euler=rotation), unit_pose()))
                        center, extent = get_center_extent(body)
                        surf_aabb = get_aabb(surface)
                        z = (surf_aabb[1] + extent / 2.)[2] + 1e-3
                        point = np.array([x, y, z]) + (get_point(body) - center)
                        pose = multiply(PoseUtils(point, rotation), unit_pose())
                        set_pose(body, pose)
                        p = pr2_primitives.Pose(body, pose, surface)
                        p.assign()
                        current_obstacles = [obst for obst in obstacles if obst not in {body, surface}]
                        collision_list = [pairwise_collision(body, obst) for obst in current_obstacles]
                        if any(collision_list):
                            # print("Collision with", [obst.name for idx, obst in enumerate(current_obstacles) if collision_list[idx]])
                            collision_str = '\n'.join([f'{Object.from_value(obst).pddl}: {get_aabb(obst)}' for idx, obst in enumerate(current_obstacles) if collision_list[idx]])
                            prompt += (
                                f"Pose {pose_name} = {value} for object {body_name} on surface {surf_name} is in collision with the following "
                                f"fixed obstacles:\n\n```\n{collision_str}\n```\n\n"
                            )
                            llm_info['num_samples_in_collision'] += 1
                            continue
                        new_body_aabb = get_aabb(body)
                        # print(f"Original aabb was {original_body_aabb}, new aabb is {new_body_aabb}")
                        if not aabb_contains_aabb(aabb2d_from_aabb(new_body_aabb), aabb2d_from_aabb(surf_aabb)):
                            # print(f"Pose {pose_name} = {value} for object {body_name} on surface {surf_name} is not stable.")
                            prompt += (
                                f"Pose {pose_name} = {value} for object {body_name} on surface {surf_name} is not stable, "
                                f"as the 2D AABB of the object ({aabb2d_from_aabb(new_body_aabb)}) is not contained in the 2D AABB "
                                f"of the surface ({aabb2d_from_aabb(surf_aabb)}). Make sure your poses are in the format `(<x> <y> <z> <theta>)`.\n"
                            )
                            llm_info['num_samples_not_stable'] += 1
                            continue
                        # Found a valid pose. 
                        llm_sampled_poses[pose_name, body, surface] = p
                        break
                    else:
                        # No valid poses found for this object, so re-prompt the LLM.
                        llm_sampled_poses[pose_name, body, surface] = None
                        llm_info['local_sampling_failures'] += 1
                # If we have sampled poses for all objects, return the action instances and the poses.
                if all(p is not None for p in llm_sampled_poses.values()):
                    # print(llm_sampled_poses)
                    new_pose_objects = {}
                    for (pose_name, body, surface), pose in llm_sampled_poses.items():
                        new_obj = Object(pose, name=repr(pose))
                        new_pose_objects[pose_name, body, surface] = new_obj
                        # print(pose_name, new_obj, new_obj.value, type(new_obj.value), new_obj.pddl)
                    replace_optimistic_objects(instantiated, action_instances, new_pose_objects)
                    print("Action instances inside solve_llm_from_pddlstream_task:")
                    for action in action_instances:
                        print(action)
                    return action_instances, llm_sampled_poses, chat
                
                ## Remove below, since now we incrementally update the prompt with each failure
                # invalid_object_poses = [name for (name, _, _), pose in llm_sampled_poses.items() if pose is None]
                # invalid_object_poses_str = ', '.join(invalid_object_poses)
                # prompt = (
                #     f"The poses you provided for the following objects are in collision: `{invalid_object_poses_str}`. "
                # )
                print()
                prompt += "Please provide a plan and poses that achieves the goal from the initial state, ensuring that all poses are valid and stable."
                print(prompt)
        else:
            invalid_actions = [name for name in plan_lines if name not in action_from_name]
            invalid_actions_str = ' , '.join(invalid_actions)
            prompt = (
                f"The plan you provided contains the following invalid actions: `{invalid_actions_str}`. "
                f"Recall that you are only allowed to use valid actions from the following list: \n\n"
                f"```\n{instantiated_actions_str}\n```\n\nPlease provide a plan that achieves the goal from the initial state "
                "using only the valid actions and adhering to the response format."
            )
            llm_info['num_invalid_actions'] += 1
            llm_info['pddl_failures'] += 1
            print()
            print(prompt)

    return None, None, None

def solve_llm_from_pddl_task(instantiated, cost_from_action, action_from_name, objects, thinking_llm=False,     
                             chat=None, error_str=None, error_str_backtracking=None, llm_info=None,
                             is_timeout_fn=lambda: False):
    ''' 
    Call an LLM as a replacement to FD to solve the instantiated task.
    '''
    assert llm_info is not None, "llm_info must be provided to track LLM usage statistics."
    if error_str_backtracking is not None:
        llm_info['num_backtracking_failures'] += 1

    geminiApiKey = os.environ["GEMINI_API_KEY"]

    # genai.configure(api_key=geminiApiKey)
    # cfg = genai.types.GenerationConfig(max_output_tokens=66667)
    if chat is None:
        chat_id = max(llm_info['chat_histories'].keys(), default=-1) + 1
        llm_info['chat_types'][chat_id] = 'pddl'
        client = genai.Client(api_key=geminiApiKey)
        # model_info = client.models.get(model="gemini-2.0-flash")
        # print(f"{model_info.input_token_limit=}")
        # print(f"{model_info.output_token_limit=}")
        sys_msg, prompt, instantiated_actions_str = get_llm_prompts_from_pddl(instantiated, cost_from_action, objects, error_str_backtracking)
        print(sys_msg)
        print()
        print(prompt)

        if thinking_llm:
            thinking_config = types.ThinkingConfig(     # Uses default thinking budget
                include_thoughts=True
            )
            generation_config = types.GenerationConfig(
                seed=llm_info['seed']
            )
            model = "gemini-2.5-flash"  # Use the latest model with thinking capabilities
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    thinking_config=thinking_config,
                    seed=llm_info['seed']
                )
            )
        else:
            thinking_config = types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,  # Set to 0 to disable thinking
            )
            model = "gemini-2.5-flash"  # Use the standard model without thinking capabilities
            chat = client.chats.create(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=sys_msg,
                    thinking_config=thinking_config,
                    seed=llm_info['seed']
                )
            )
        chat.chat_id = chat_id
    else:
        # The prompt should be the failure message, which must be a failed precondition or goal
        assert error_str is not None
        prompt = (
            f"The plan you provided is not valid because: `{error_str}`. "
            f"Recall that the action preconditions and the goal *must* be satisfied.\n\nPlease provide a plan that achieves the goal from the initial state "
            "ensuring that action preconditions are satisfied and adhering to the response format."
        )
        print(prompt)
        _, _, instantiated_actions_str = get_llm_prompts_from_pddl(instantiated, cost_from_action, objects)

    while not is_timeout_fn():
        start_time = time.time()
        response = chat.send_message(prompt)
        end_time = time.time()
        print(response.usage_metadata)
        print(response.text)
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        thinking_tokens = response.usage_metadata.thoughts_token_count or 0
        llm_info['pddl_input_tokens'] += input_tokens
        llm_info['pddl_output_tokens'] += output_tokens
        llm_info['pddl_thinking_tokens'] += thinking_tokens
        llm_info['chat_histories'][chat.chat_id] = chat.get_history()
        llm_info['pddl_time'] += (end_time - start_time)
        for part in response.candidates[0].content.parts:
            if part and part.thought:  # show thoughts
                print(part.text)
        if response.text is None:
            print("No response text found.")
            llm_info['num_no_plans_returned'] += 1
            return None, None
        # Find the plan in the response text and return as a list of action names
        plan_start = response.text.find("--Plan--:")
        if plan_start == -1:
            print("No plan found in the response.")
            llm_info['num_no_plans_returned'] += 1
            return None, None
        plan_text = response.text[plan_start + len("--Plan--:"):].strip()
        plan_lines = plan_text.split('\n')
        # Remove any text beyond ')' in each line
        plan_lines = [line.split(')')[0] + ')' for line in plan_lines if line.strip() != '']
        if len(plan_lines) == 0:
            print("No plan found in the response.")
            llm_info['num_no_plans_returned'] += 1
            return None, None
        if all(name in action_from_name for name in plan_lines):
            action_instances = [action_from_name[name] for name in plan_lines]
            return action_instances, chat
        invalid_actions = [name for name in plan_lines if name not in action_from_name]
        invalid_actions_str = ', '.join(invalid_actions)
        prompt = (
            f"The plan you provided contains the following invalid actions: `{invalid_actions_str}`. "
            f"Recall that you are only allowed to use valid actions from the following list: \n\n"
            f"```\n{instantiated_actions_str}\n```\n\nPlease provide a plan that achieves the goal from the initial state "
            "using only the valid actions and adhering to the response format."
        )
        llm_info['num_invalid_actions'] += 1
        llm_info['pddl_failures'] += 1
        print()
        print(prompt)

    return None, None


def solve_optimistic_sequential(domain, stream_domain, applied_results, all_results,
                                opt_evaluations, node_from_atom, goal_expression,
                                effort_weight, debug=False,
                                pddl_llm=False, thinking_llm=False, integrated_llm=False,
                                chat=None, error_str=None, error_str_backtracking=None,
                                pybullet_obstacles=None, pybullet_rovers=None, llm_info=None,
                                is_timeout_fn=lambda: False, **kwargs):
    #print(sorted(map(fact_from_evaluation, opt_evaluations)))
    rename_actions = RENAME_ACTIONS and not (pddl_llm or integrated_llm)
    temporal_plan = None
    objects = objects_from_evaluations(opt_evaluations)
    problem = get_problem(opt_evaluations, goal_expression, stream_domain)  # begin_metric
    with Verbose(verbose=debug):
        task = task_from_domain_problem(stream_domain, problem)
        instantiated = instantiate_task(task)
    
    if instantiated is None:
        # print("Inside solve_optimistic_sequential, instantiated is None")
        return instantiated, None, temporal_plan, INF, {}, None
    
    cost_from_action = {action: action.cost for action in instantiated.actions}
    add_stream_efforts(node_from_atom, instantiated, effort_weight)
    cost_effort_from_action = {action: action.cost for action in instantiated.actions}
    if using_optimizers(applied_results):
        add_optimizer_effects(instantiated, node_from_atom)
        # TODO: reachieve=False when using optimizers or should add applied facts
        instantiate_optimizer_axioms(instantiated, domain, all_results)
    action_from_name = rename_instantiated_actions(instantiated, rename_actions)
    if integrated_llm:
        action_instances, sampled_poses, chat = solve_llm_from_pddlstream_task(
            instantiated, cost_effort_from_action, action_from_name, objects, pybullet_obstacles,
            pybullet_rovers, thinking_llm=thinking_llm, chat=chat, error_str=error_str, 
            error_str_backtracking=error_str_backtracking, llm_info=llm_info, is_timeout_fn=is_timeout_fn)
    elif pddl_llm:
        action_instances, chat = solve_llm_from_pddl_task(instantiated, cost_effort_from_action, action_from_name, objects, thinking_llm=thinking_llm, chat=chat, error_str=error_str,
        error_str_backtracking= error_str_backtracking, llm_info=llm_info, is_timeout_fn=is_timeout_fn)
        sampled_poses = {}
    else:
        # TODO: the action unsatisfiable conditions are pruned
        with Verbose(debug):
            sas_task = sas_from_instantiated(instantiated)
            #sas_task.metric = task.use_min_cost_metric
            sas_task.metric = True

        # TODO: apply renaming to hierarchy as well
        # solve_from_task | serialized_solve_from_task | abstrips_solve_from_task | abstrips_solve_from_task_sequential

        renamed_plan, _ = solve_from_task(sas_task, debug=debug, **kwargs)
        if renamed_plan is None:
            return instantiated, None, temporal_plan, INF, {}, None

        action_instances = [action_from_name[name if rename_actions else '({} {})'.format(name, ' '.join(args))]
                            for name, args in renamed_plan]

        sampled_poses = {}
        
        
        print("--Plan--:")
        for action in action_instances:
            print(action.name, pddl_from_instance(action), transform_action_args(pddl_from_instance(action), obj_from_pddl))
        chat = None
    cost = get_plan_cost(action_instances, cost_from_action)

    return instantiated, action_instances, temporal_plan, cost, sampled_poses, chat

##################################################

def plan_streams(evaluations, goal_expression, domain, all_results, negative, effort_weight, max_effort,
                 simultaneous=False, reachieve=True, replan_actions=set(), pddl_problem=None, 
                 is_timeout_fn=lambda: False, llm_info=None, **kwargs):
    # TODO: alternatively could translate with stream actions on real opt_state and just discard them
    # TODO: only consider axioms that have stream conditions?
    #reachieve = reachieve and not using_optimizers(all_results)
    #for i, result in enumerate(all_results):
    #    print(i, result, result.get_effort())'
    prev_evaluations = evaluations
    applied_results, deferred_results = partition_results(
        evaluations, all_results, apply_now=lambda r: not (simultaneous or r.external.info.simultaneous))
    stream_domain, deferred_from_name = add_stream_actions(domain, deferred_results)

    if reachieve and not using_optimizers(all_results):
        achieved_results = {n.result for n in evaluations.values() if isinstance(n.result, Result)}
        init_evaluations = {e for e, n in evaluations.items() if n.result not in achieved_results}
        applied_results = achieved_results | set(applied_results)
        evaluations = init_evaluations # For clarity 

    # TODO: could iteratively increase max_effort
    node_from_atom = get_achieving_streams(evaluations, applied_results, # TODO: apply to all_results?
                                           max_effort=max_effort)
    opt_evaluations = {evaluation_from_fact(f): n.result for f, n in node_from_atom.items()}
    if UNIVERSAL_TO_CONDITIONAL or using_optimizers(all_results):
        goal_expression = add_unsatisfiable_to_goal(stream_domain, goal_expression)

    temporal = isinstance(stream_domain, SimplifiedDomain)
    optimistic_fn = solve_optimistic_temporal if temporal else solve_optimistic_sequential
    while not is_timeout_fn():
        # print("Entering while True loop in plan_streams")
        instantiated, action_instances, temporal_plan, cost, sampled_poses, chat = optimistic_fn(
            domain, stream_domain, applied_results, all_results, opt_evaluations,
            node_from_atom, goal_expression, effort_weight, llm_info=llm_info, is_timeout_fn=is_timeout_fn, **kwargs)
        if action_instances is None:
            # print("Action instances is none, returning FAILED FAILED cost")
            return OptSolution(FAILED, FAILED, cost)

        try: 
            action_instances, axiom_plans = recover_axioms_plans(instantiated, action_instances)
        except RuntimeError as err:
            # print("Got a runtime error in recover_axiom_plans:", err)
            if ('pddl_llm' in kwargs and kwargs['pddl_llm']) or ('integrated_llm' in kwargs and kwargs['integrated_llm']):
                # This means the LLM did not reach the goal
                # error_str = str(err).replace("Fact", "Goal fact").replace("achievable", "satisfied")
                error_str = str(err).replace("achievable", "satisfied")     # It could also be a non-goal fact (e.g., an axiom that is in a precondition)
                kwargs['chat'] = chat
                kwargs['error_str'] = error_str
                llm_info['num_axioms_not_achieved'] += 1
                llm_info['pddl_failures'] += 1
                continue
            else:
                raise
       # My goal is to:
            # Add the "certified" evaluations from the sample_pose stream for any sampled pose
                # Those are 'pose' and 'supported'
            # Remove the 'pose' and 'supported' evaluations from the opt_evaluations
            # Replace the arg in the non-certified evaluations like kin and traj and such in opt_evaluations
            # Create nodes for certified facts with no effort and result=None
            # Remove the original certified facts from the node_from_atom map
            # Replace the args in the node_from_atom with the new pose objects

        '''
        achieved_results = {n.result for n in evaluations.values() if isinstance(n.result, Result)}
        init_evaluations = {e for e, n in evaluations.items() if n.result not in achieved_results}
        applied_results = achieved_results | set(applied_results)
        evaluations = init_evaluations # For clarity 

        # TODO: could iteratively increase max_effort
        node_from_atom = get_achieving_streams(evaluations, applied_results, # TODO: apply to all_results?
                                            max_effort=max_effort)
        opt_evaluations = {evaluation_from_fact(f): n.result for f, n in node_from_atom.items()}

        --------
        llm_sampled_poses[pose_name, body_name, surf_name] = p

        ''' 
        new_facts = []
        init = instantiated.task.init
        for (pose_name, body, surf), pose in sampled_poses.items():
            # Create a new evaluation for the 'pose' and 'supported' facts
            pose_fact = ('Pose', body, pose)
            supported_fact = ('Supported', body, pose, surf)
            # print("Adding pose fact:", pose_fact)
            # print("Adding supported fact:", supported_fact)
            new_facts.append(pose_fact)
            new_facts.append(supported_fact)
            for fact in init:
                if isinstance(fact, Assign):
                    continue
                if fact.predicate == 'kin':
                    fact_args_set = set(fact.args)
                    pose_body_set = set([pose_name, pddl_from_object(Object._obj_from_value[body])])
                    intersection = fact_args_set & pose_body_set
                    if len(intersection) == 2:  # It means that the fact is about this pose
                        new_args = []
                        for arg in fact.args:
                            if arg == pose_name:
                                new_args.append(pose)
                            else:
                                # new_args.append(obj_from_pddl(arg))
                                new_args.append(Object.from_value(arg))
                        kin_fact = ('Kin', *new_args)
                        # print("Adding kin fact:", kin_fact)
                        new_facts.append(kin_fact)
       
        ############ Add the new facts to pddl_problem.init
        facts_for_pddl_problem = []
        facts_in_pddl_problem = set(pddl_problem.init)
        for fact in init:
            if isinstance(fact, Assign):
                continue
            pred = fact.predicate.title()
            if pred == 'Canmove': pred = 'CanMove'
            elif pred == 'Bconf': pred = 'BConf'
            elif pred == 'Atbconf': pred = 'AtBConf'
            elif pred == 'Aconf': pred = 'AConf'
            elif pred == 'Handempty': pred = 'HandEmpty'
            elif pred == 'Ataconf': pred = 'AtAConf'
            elif pred == 'Atpose': pred = 'AtPose'
            elif pred == 'Notcausesfailure': pred = 'NotCausesFailure'
            elif pred == 'Notcausesfailuresolo': pred = 'NotCausesFailureSolo'
            elif pred == 'Btraj': pred = 'BTraj'
            elif pred == 'Basemotion': pred = 'BaseMotion'
            elif pred == 'Atraj': pred = 'ATraj'
            elif pred == '=': continue
            elif pred == 'Identical': continue
            if any(isinstance(obj_from_pddl(arg).value, (SharedOptValue, UniqueOptValue)) for arg in fact.args): continue
            new_fact = (pred, *map(lambda x: obj_from_pddl(x).value, fact.args))
            if new_fact not in facts_in_pddl_problem:
                facts_for_pddl_problem.append(new_fact)
        if pddl_problem is not None:
            pddl_problem.init.extend(facts_for_pddl_problem)

        new_evaluations = evaluations_from_init(new_facts)
        prev_evaluations |= new_evaluations

        new_results = []
        sampled_optimistic_poses = {(OptimisticObject.from_name(pose_name), Object.from_value(body), Object.from_value(surface)): Object.from_value(pose) for (pose_name, body, surface), pose in sampled_poses.items()}
        for result in all_results:
            # print("Looking into result:", result)
            if isinstance(result, FunctionResult):
                # print("\tIt is a function result, keeping it as is")
                new_results.append(result)
            else:
                # If any input object is a sampled pose, rebind
                input_objects_set = set(result.instance.input_objects)
                for opt_pose, body, surface in sampled_optimistic_poses:
                    pose_body_set = set([opt_pose, body])
                    intersection = input_objects_set & pose_body_set
                    if len(intersection) == 2:  # It means that the result is about this pose
                        new_input_objects = {}
                        for obj in result.instance.input_objects:
                            if obj == opt_pose:
                                new_input_objects[obj] = sampled_optimistic_poses[(opt_pose, body, surface)]
                        new_result = result.remap_inputs(new_input_objects)
                        new_results.append(new_result)
                        # print("\tSome input object is a sampled pose, rebinding to", new_result)
                else:
                    # If no output object is a sampled pose, keep the original result
                    for obj in result.output_objects:
                        if obj in sampled_optimistic_poses:
                            # print("\tSome output object is a sampled pose, skipping this result")
                            break
                    else:
                        # print("\tNo sampled pose in input or output objects, keeping the original result")
                        new_results.append(result)
        
        all_results = new_results

        applied_results, deferred_results = partition_results(
            prev_evaluations, all_results, apply_now=lambda r: not (simultaneous or r.external.info.simultaneous))
        stream_domain, deferred_from_name = add_stream_actions(domain, deferred_results)
        achieved_results = {n.result for n in prev_evaluations.values() if isinstance(n.result, Result)}
        init_evaluations = {e for e, n in prev_evaluations.items() if n.result not in achieved_results}
        applied_results = achieved_results | set(applied_results)
        evaluations = init_evaluations # For clarity
        node_from_atom = get_achieving_streams(evaluations, applied_results, # TODO: apply to all_results?
                                            max_effort=max_effort)
        opt_evaluations = {evaluation_from_fact(f): n.result for f, n in node_from_atom.items()}

        # evaluations doesn't contain anything about the new pose object
        # But opt_evaluations does contain facts about the optimistic pose
        # not sure how to handle this
        # print("All results:")
        # for result in all_results:
        #     print("\t", result, type(result), [type(obj) for obj in result.output_objects]  if not isinstance(result, FunctionResult) else '')
        # print("Evaluations:")
        # for eval in evaluations:
        #     print("\t",eval)
        # print("Opt evaluations:")
        # for eval in opt_evaluations:
        #     print("\t", eval)
        # # Node from atom is the other object that contains stuff about the optimistic pose
        # # I'll need to add 'pose' and 'supported' for each pose we've created, and replace
        # # the evaluation 
        # print("Node from atom:")
        # for k, v in node_from_atom.items():
        #     print("\t", k,  v, v.result.instance.get_domain() if v.result is not None else None)
       
        # # TODO: extract out the minimum set of conditional effects that are actually required
        # #simplify_conditional_effects(instantiated.task, action_instances)
        stream_plan, action_instances = recover_simultaneous(
            applied_results, negative, deferred_from_name, action_instances)
        # print("Stream plan:")
        # for stream in stream_plan:
        #     print(stream.name, type(stream), stream.optimistic, stream.output_objects if not isinstance(stream, FunctionResult) else None, [type(obj) for obj in stream.output_objects] if not isinstance(stream, FunctionResult) else None)
        # print("Goal expression:", goal_expression)
        # print("Stream domain:")
        # for elem in domain:
        #     print(elem)
        # print("Action instances")
        # for action in action_instances:
        #     print(action.name)
        #     for attr in dir(action):
        #         if not attr.startswith('_'):
        #             if isinstance(getattr(action, attr), dict):
        #                 for k, v in getattr(action, attr).items():
        #                     print(f"\t{attr} {k} ({type(k)}): {v} ({type(v)})")
        #             elif isinstance(getattr(action, attr), (list, tuple)):
        #                 for elem in getattr(action, attr):
        #                     print(f"\t{attr}: {elem} ({type(elem)})")
        #             else:
        #                 print("\t", attr, getattr(action, attr), type(getattr(action, attr)))
        # print("Axiom plan")
        # for axiom_plan in axiom_plans:
        #     for axiom in axiom_plan:
        #         for attr in dir(axiom):
        #             if not attr.startswith('_'):
        #                 if isinstance(getattr(axiom, attr), dict):
        #                     for k, v in getattr(axiom, attr).items():
        #                         print(f"\t{attr} {k} ({type(k)}): {v} ({type(v)})")
        #                 elif isinstance(getattr(axiom, attr), (list, tuple)):
        #                     for elem in getattr(axiom, attr):
        #                         print(f"\t{attr}: {elem} ({type(elem)})")
        #                 else:
        #                     print("\t", attr, getattr(axiom, attr), type(getattr(axiom, attr)))
        # print("Negative:", negative)


        action_plan = transform_plan_args(map(pddl_from_instance, action_instances), obj_from_pddl)
        replan_step = min([step+1 for step, action in enumerate(action_plan)
                        if action.name in replan_actions] or [len(action_plan)+1]) # step after action application

        try:
            # print("Node from atom:", node_from_atom)
            stream_plan, opt_plan = recover_stream_plan(evaluations, stream_plan, opt_evaluations, goal_expression, stream_domain,
                node_from_atom, action_instances, axiom_plans, negative, replan_step)
            
            #######
            # print("Length of stream plan:", len(stream_plan))
            # print(stream_plan)
            # print("Length of action plan:", len(opt_plan.action_plan))
            # for action in opt_plan.action_plan:
            #     print(action, action.name, type(action))
            break
        except RuntimeError as err:
            # print("Got a runtime error in recover_stream_plan:", err)
            if ('pddl_llm' in kwargs and kwargs['pddl_llm']) or ('integrated_llm' in kwargs and kwargs['integrated_llm']):
                error_str = str(err).replace("Preimage", "Precondition").replace("achievable", "satisfied")
                # Rename the objects in the error to their PDDL names
                # Find things in parenthesis and replace them with their PDDL names
                obj_str = error_str[error_str.rfind('(')+1:error_str.rfind(')')]
                obj_names = obj_str.split(', ')[1:]
                pddl_names = []
                # Recreate new_pose_objects dictionary:
                new_pose_objects = {}
                for (pose_name, _, _), pose in sampled_poses.items():
                    pose_obj_str = str(Object._obj_from_value[pose])
                    new_pose_objects[pose_obj_str] = pose_name
                # print("new_pose_objects:", new_pose_objects)
                for name in obj_names:
                    # print("Need to replace", name, f"({type(name)}) with", pddl_from_object(obj_from_value_str(name)))
                    if name in new_pose_objects:
                        pddl_name = new_pose_objects[name]
                    else:
                        pddl_name = pddl_from_object(obj_from_value_str(name))
                    pddl_names.append(pddl_name)
                new_obj_str = ", ".join([obj_str.split(', ')[0]] + pddl_names)
                error_str = error_str.replace(obj_str, new_obj_str)
                kwargs['chat'] = chat
                kwargs['error_str'] = error_str
                llm_info['num_preimages_not_achieved'] += 1
                llm_info['pddl_failures'] += 1
                continue
            else:
                raise
    else:
        # Timed out
        return OptSolution(FAILED, FAILED, float('inf'))
    
    if temporal_plan is not None:
        # TODO: handle deferred streams
        assert all(isinstance(action, Action) for action in opt_plan.action_plan)
        opt_plan.action_plan[:] = temporal_plan
    return OptSolution(stream_plan, opt_plan, cost)
