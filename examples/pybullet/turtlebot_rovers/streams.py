from __future__ import print_function

import numpy as np
import time
import os

from examples.pybullet.utils.pybullet_tools.pr2_primitives import Conf, Trajectory, create_trajectory, Command
from examples.pybullet.utils.pybullet_tools.pr2_utils import visible_base_generator, get_detection_cone
from examples.pybullet.utils.pybullet_tools.utils import get_point, get_custom_limits, all_between, pairwise_collision, \
    plan_joint_motion, get_sample_fn, get_distance_fn, get_collision_fn, check_initial_end, is_placement, \
    MAX_DISTANCE, get_extend_fn, wait_for_user, create_mesh, set_pose, get_link_pose, link_from_name, \
    remove_body, create_cylinder, get_distance, point_from_pose, Pose, Point, multiply, get_visual_data, get_pose, \
    wait_for_duration, create_body, visual_shape_from_data, LockRenderer, plan_nonholonomic_motion, create_attachment, \
    pose_from_pose2d, wait_if_gui, child_link_from_joint, get_link_name, Attachment, get_aabb, wrap_angle
from examples.pybullet.turtlebot_rovers.problems import get_base_joints, KINECT_FRAME
from pddlstream.language.constants import Output

from google import genai
from google.genai import types

VIS_RANGE = 2
COM_RANGE = 2*VIS_RANGE

class Ray(Command):
    _duration = 1.0
    def __init__(self, body, start, end):
        self.body = body
        self.start = start
        self.end = end
        self.pose = get_pose(self.body)
        self.visual_data = get_visual_data(self.body)
    def apply(self, state, **kwargs):
        print(self.visual_data)
        with LockRenderer():
            visual_id = visual_shape_from_data(self.visual_data[0]) # TODO: TypeError: argument 5 must be str, not bytes
            cone = create_body(visual_id=visual_id)
            #cone = create_mesh(mesh, color=(0, 1, 0, 0.5))
            set_pose(cone, self.pose)
        wait_for_duration(self._duration)
        with LockRenderer():
            remove_body(cone)
            wait_for_duration(1e-2)
        wait_for_duration(self._duration)
        # TODO: set to transparent before removing
        yield
    def __repr__(self):
        return '{}->{}'.format(self.start, self.end)


def get_reachable_test(problem, iterations=10, **kwargs):
    initial_confs = {rover: Conf(rover, get_base_joints(rover))
                     for rover in problem.rovers}
    # TODO: restarts -> max_restarts
    motion_fn = get_motion_fn(problem, restarts=0, max_iterations=iterations, smooth=0, **kwargs)
    def test(rover, bq):
        bq0 = initial_confs[rover]
        result = motion_fn(rover, bq0, bq)
        return result is not None
    return test


def get_cfree_ray_test(problem, collisions=True):
    def test(ray, rover, conf):
        if not collisions or (rover == ray.start) or (rover == ray.end):
            return True
        conf.assign()
        collision = pairwise_collision(ray.body, rover)
        #if collision:
        #    wait_for_user()
        return not collision
    return test


def get_inv_vis_gen(problem, use_cone=True, max_attempts=25, max_range=VIS_RANGE,
                    custom_limits={}, collisions=True, 
                    llm_sampler=False, thinking_llm=False, 
                    llm_info=None, **kwargs):
    base_range = (0, max_range)
    obstacles = problem.fixed if collisions else []
    reachable_test = get_reachable_test(problem, custom_limits=custom_limits, collisions=collisions, **kwargs)

    if llm_sampler:
        assert llm_info is not None, "llm_info must be provided to track LLM usage statistics."
        obstacle_aabbs = [get_aabb(o) for o in obstacles]
        obstacle_str = "```\n" + \
            "\n".join([f"{idx}: (({aabb[0][0]}, {aabb[0][1]}), ({aabb[1][0]}, {aabb[1][1]}))" 
                       for idx, aabb in enumerate(obstacle_aabbs)]) + "\n```"

        sys_msg = (
            "You are highly skilled at geometric reasoning. You do an excellent job of finding "
            "base positions for a robot to view a target object while avoiding collisions and "
            "occlusions. You are given a target point , and you must find a "
            f"base position (x y) within *max_range={max_range}* distance from the target point. The robot's orientation "
            "theta will be set automatically to face the target point. You are not given the full "
            "geometry of the scene, but you will be given feedback of whether a proposed position collides "
            "with any of the fixed obstacles in the scene and has line-of-sight to the target. If you receive a second request for "
            "the same target point, it may be because the previous position collided with some non-fixed "
            "obstacle, which you must use to learn about the geometry of the scene. You may reuse previous "
            "positions if they appear valid, but if a position fails multiple times, it is likely that there is "
            "a collision or an occlusion that you are not aware of, and you should try a different position.\n\n"
            "The geometry of the fixed obstacles, described as a list of axis-aligned bounding boxes (AABBs), is:"
            f"{obstacle_str}\n\n"
            "You never give up. No matter how many times you fail to provide a position, or how many valid positions "
            "you have already provided, you will always try to find a new position.\n\n"
            f"Please return {max_attempts} positions for the robot base that are within the max distance {max_range} from the target point. "
            "A good strategy is to generate a diverse set that covers many positions that are likely collision- "
            "and occlusion-free. You can first describe your knowledge of the scene and of geometry to help you "
            "come up with a position. Please provide your output in the following format (excluding the angle "
            "brackets, which are just for illustration purposes). Be sure to include the parentheses '(' and ')'. "
            "Do not bold or italicize or otherwise apply any extra formatting to the position text. Do not provide "
            "any numbers for positions in the list, or any reasoning or comments for each position below the  'Positions:' "
            "heading. Use the following format for your response:\n\n"
            "```\n<Explanation of the scene + your reasoning (optional)\nPositions:\n"
            f"(<x_1> <y_1>)\n(<x_2> <y_2>)\n...\n(<x_{max_attempts}> <y_{max_attempts}>)\n```\n\n"
        )
        print(sys_msg)
        failure_history = []
        success_history = []
        geminiApiKey = os.environ['GEMINI_API_KEY']
        client = genai.Client(api_key=geminiApiKey)
        if thinking_llm:
            thinking_config = types.ThinkingConfig(
                include_thoughts=True,
            )
        else:
            thinking_config = types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,  # No thinking budget for Gemini 2.5 Flash
            )
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=sys_msg,
                thinking_config=thinking_config,
                seed=llm_info['seed']
            )
        )
        chat_id = max(llm_info['chat_histories'].keys(), default=-1) + 1
        chat.chat_id = chat_id
        llm_info['chat_types'][chat_id] = 'sampling'

    def gen(rover, objective):
        base_joints = get_base_joints(rover)
        target_point = get_point(objective)
        base_generator = visible_base_generator(rover, target_point, base_range)
        lower_limits, upper_limits = get_custom_limits(rover, base_joints, custom_limits)
        attempts = 0
        
        if llm_sampler:
            prompt = ""
            if len(success_history) > 0:
                prompt += (
                    "The following successes were found from the previous request. These were "
                    "positions that did not collide with fixed obstacles. "
                    "You may use these as positive examples for future requests:\n\n"
                    "```\n" + "\n".join(success_history) + +"\n```\n\n"
                )
            if len(failure_history) > 0:
                prompt += (
                    "The following failures were found from the previous request. These were "
                    "positions that either collided with fixed obstacles or did not have line-of-sight "
                    "to the target point. You may use these as negative examples for future requests:\n\n"
                    "```\n" + "\n".join(failure_history) + "\n```\n\n"
                )
            if lower_limits and upper_limits:
                prompt += (
                    "The robot can only be placed within the following limits for (x, y, theta):\n"
                    f"x: [{lower_limits[0]}, {upper_limits[0]}], "
                    f"y: [{lower_limits[1]}, {upper_limits[1]}], "
                    f"theta: [{lower_limits[2]}, {upper_limits[2]}]\n\n"
                )
            prompt += (
                "New request:\n\nTarget point: {target_point}. Please provide a valid placement for the "
                f"robot base (x y) within the max distance {max_range} from the target point. "
            )
            print(prompt)
            start_time = time.time()
            response = chat.send_message(prompt)
            end_time = time.time()
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            thinking_tokens = response.usage_metadata.thoughts_token_count or 0
            llm_info['sampling_input_tokens'] += input_tokens
            llm_info['sampling_output_tokens'] += output_tokens
            llm_info['sampling_thinking_tokens'] += thinking_tokens
            llm_info['sampling_time'] += (end_time - start_time)
            llm_info['chat_histories'][chat_id] = chat.get_history()
            print(response.text)
            positions_start = response.text.find("Positions:")
            if positions_start == -1:
                print("No positions found in response")
                llm_info['num_no_samples_returned'] += 1
                yield None # should I yield or return? They are conceptually different (one says "current sample unavailable" the other says "ran out") but PDDLStream treats them equally
                # My reasoning above was false: return None cannot continue ever. yield None is the correct choice
            positions_text = response.text[positions_start + len("Positions:"):].strip()
            position_lines = positions_text.split("\n")
            print(f"Found {len(position_lines)} position lines in response")
            for line in position_lines:
                try:
                    line = line.strip('() ')
                    x, y = map(float, line.split())
                except ValueError:
                    try:
                        x, y = map(float, line.split(','))
                        print("Note: parsed position line with comma instead of space")
                    except ValueError as err:
                        print(f"Invalid position line: {line}")
                        print(f"Error: {err}")
                        llm_info['num_samples_format_failures'] += 1
                        continue
                base_theta = np.math.atan2(target_point[1]-y, target_point[0]-x)
                base_q = np.append([x, y], wrap_angle(base_theta))
                # Check that the sample is within joint limits
                if not all_between(lower_limits, base_q, upper_limits):
                    print(f"Position ({x}, {y}) is out of bounds")
                    llm_info['num_samples_out_of_bounds'] += 1
                    failure_history.append(f"Target point: {target_point}, Proposed position: ({x}, {y}) -- out of bounds")
                    continue
                bq = Conf(rover, base_joints, base_q)
                bq.assign()
                # Check that the sample does not collide with fixed obstacles
                collision_list = [pairwise_collision(rover, obst) for obst in obstacles]
                if any(collision_list):
                    collision_aabbs = [get_aabb(obst) for idx, obst in enumerate(obstacles) if collision_list[idx]]
                    collision_str = " collides with:\n\n```\n" + \
                        "\n".join([f"{idx}: (({aabb[0][0]}, {aabb[0][1]}), ({aabb[1][0]}, {aabb[1][1]}))" 
                                   for idx, aabb in enumerate(collision_aabbs)]) + "\n```"
                    print(f"Position ({x}, {y}) collides with fixed obstacles")
                    failure_history.append(f"Target point: {target_point}, Proposed position: ({x}, {y}) -- {collision_str}")
                    llm_info['num_samples_in_collision'] += 1
                    continue
                # Check that the sample has line-of-sight to the target point
                link_pose = get_link_pose(rover, link_from_name(rover, KINECT_FRAME))
                if use_cone:
                    mesh, _ = get_detection_cone(rover, objective, camera_link=KINECT_FRAME, depth=max_range)
                    if mesh is None:
                        print(f"No mesh for detection cone")
                        llm_info['num_samples_not_visible'] += 1
                        failure_history.append(f"Target point: {target_point}, Proposed position: ({x}, {y}) -- no line of sight to target")
                        continue
                    cone = create_mesh(mesh, color=(0, 1, 0, 0.5))
                    local_pose = Pose()
                else:
                    distance = get_distance(point_from_pose(link_pose), target_point)
                    if max_range < distance:
                        print(f"Position ({x}, {y}) is out of range (distance {distance})")
                        llm_info['num_samples_out_of_range'] += 1
                        failure_history.append(f"Target point: {target_point}, Proposed position: ({x}, {y}) -- farther than max_range={max_range}")
                        continue
                    cone = create_cylinder(radius=0.01, height=distance, color=(0, 0, 1, 0.5))
                    local_pose = Pose(Point(z=distance/2.))
                set_pose(cone, multiply(link_pose, local_pose))
                
                if any(pairwise_collision(cone, b) for b in obstacles
                       if b != objective and not is_placement(objective, b)):
                    remove_body(cone)
                    print(f"Position ({x}, {y}) does not have line of sight to target (occluded by obstacles)")
                    llm_info['num_samples_not_visible'] += 1
                    failure_history.append(f"Target point: {target_point}, Proposed position: ({x}, {y}) -- no line of sight to target due to occlusion")
                    continue
                if not reachable_test(rover, bq):
                    print(f"Position ({x}, {y}) is not reachable")
                    llm_info['num_samples_not_reachable'] += 1
                    failure_history.append(f"Target point: {target_point}, Proposed position: ({x}, {y}) -- not reachable for robot base")
                    continue
                print("Found valid placement, will yield: ", line.strip())
                llm_info['num_samples_used'] += 1
                y = Ray(cone, rover, objective)
                yield Output(bq, y)
            llm_info['local_sampling_failures'] += 1
            print("LLM sample failed:")
            print(rover, target_point, "Failed")


        else:
            while True:
                if max_attempts <= attempts:
                    attempts = 0
                    print("Ran out of attempts")
                    yield None
                attempts += 1
                base_conf = next(base_generator)
                if not all_between(lower_limits, base_conf, upper_limits):
                    print(f"Attempt {attempts} base conf out of bounts")
                    continue
                bq = Conf(rover, base_joints, base_conf)
                bq.assign()
                if any(pairwise_collision(rover, b) for b in obstacles):
                    print(f"Attempt {attempts} base conf collides with obstacles")
                    continue

                link_pose = get_link_pose(rover, link_from_name(rover, KINECT_FRAME))
                if use_cone:
                    mesh, _ = get_detection_cone(rover, objective, camera_link=KINECT_FRAME, depth=max_range)
                    if mesh is None:
                        print(f"Attempt {attempts} no mesh for detection cone")
                        continue
                    cone = create_mesh(mesh, color=(0, 1, 0, 0.5))
                    local_pose = Pose()
                else:
                    distance = get_distance(point_from_pose(link_pose), target_point)
                    if max_range < distance:
                        continue
                    cone = create_cylinder(radius=0.01, height=distance, color=(0, 0, 1, 0.5))
                    local_pose = Pose(Point(z=distance/2.))
                set_pose(cone, multiply(link_pose, local_pose))
                # TODO: colors corresponding to scanned object

                if any(pairwise_collision(cone, b) for b in obstacles
                    if b != objective and not is_placement(objective, b)):
                    # TODO: ensure that this works for the rover
                    remove_body(cone)
                    print(f"Attempt {attempts} cone collides with obstacles")
                    continue
                if not reachable_test(rover, bq):
                    print(f"Attempt {attempts} base conf not reachable")
                    continue
                print('Visibility attempts:', attempts)
                y = Ray(cone, rover, objective)
                llm_info['num_samples_used'] += 1
                yield Output(bq, y)
                #break
    return gen


def get_inv_com_gen(problem, **kwargs):
    return get_inv_vis_gen(problem, use_cone=False, max_range=COM_RANGE, **kwargs)


def get_above_gen(problem, max_attempts=1, custom_limits={}, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    reachable_test = get_reachable_test(problem, custom_limits=custom_limits, collisions=collisions, **kwargs)

    def gen(rover, rock):
        base_joints = get_base_joints(rover)
        x, y, _ = get_point(rock)
        lower_limits, upper_limits = get_custom_limits(rover, base_joints, custom_limits)
        while True:
            for _ in range(max_attempts):
                theta = np.random.uniform(-np.pi, np.pi)
                base_conf = [x, y, theta]
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(rover, base_joints, base_conf)
                bq.assign()
                if any(pairwise_collision(rover, b) for b in obstacles):
                    continue
                if not reachable_test(rover, bq):
                    continue
                yield Output(bq)
                break
            else:
                yield None
    return gen

#######################################################

def get_motion_fn(problem, custom_limits={}, collisions=True, teleport=False, holonomic=False, reversible=False, **kwargs):
    def test(rover, q1, q2, fluents=[]):
        if teleport:
            ht = Trajectory([q1, q2])
            return Output(ht)

        base_link = child_link_from_joint(q1.joints[-1])
        q1.assign()
        attachments = []
        movable = set()
        for fluent in fluents:
            predicate, args = fluent[0], fluent[1:]
            if predicate == 'AtGrasp'.lower():
                r, b, g = args
                attachments.append(Attachment(rover, base_link, g.value, b))
            elif predicate == 'AtPose'.lower():
                b, p = args
                assert b not in movable
                p.assign()
                movable.add(b)
            # elif predicate == 'AtConf'.lower():
            #     continue
            else:
                raise NotImplementedError(predicate)

        obstacles = set(problem.fixed) | movable if collisions else []
        q1.assign()
        if holonomic:
            path = plan_joint_motion(rover, q1.joints, q2.values, custom_limits=custom_limits,
                                     attachments=attachments, obstacles=obstacles, self_collisions=False, **kwargs)
        else:
            path = plan_nonholonomic_motion(rover, q1.joints, q2.values, reversible=reversible, custom_limits=custom_limits,
                                            attachments=attachments, obstacles=obstacles, self_collisions=False, **kwargs)
        if path is None:
            return None
        ht = create_trajectory(rover, q2.joints, path)
        return Output(ht)
    return test
