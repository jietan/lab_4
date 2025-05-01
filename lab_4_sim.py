#@title Import packages
import time
import numpy as np
import math
# Graphics and plotting.
import mediapy as media
# Mujoco, MJX, and Brax
import jax
from jax import numpy as jp
import argparse
import create_env
import ik
# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

N_STEPS = 1000
RENDER_EVERY = 2
fix_base = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_base', action='store_true', help='Fix the base position and orientation')
    parser.set_defaults(fix_base=False)
    return parser.parse_args()

def interpolate_triangle(t, leg_index):
    touch_down_position = np.array([0.05, 0.0, -0.14])
    stand_position_1 = np.array([0.025, 0.0, -0.14])
    stand_position_2 = np.array([0.0, 0.0, -0.14]) 
    stand_position_3 = np.array([-0.025, 0.0, -0.14])
    liftoff_position = np.array([-0.05, 0.0, -0.14])
    mid_swing_position = np.array([0.0, 0.0, -0.05])
    front_leg_offset = 0.07
    rear_leg_offset = -0.09

    # Define waypoints for each leg
    if leg_index == 0:  # Right front
        waypoints = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2, 
            stand_position_3,
            liftoff_position,
            mid_swing_position
        ]) + np.array([front_leg_offset, -0.09, 0])
    elif leg_index == 1:  # Left front
        waypoints = np.array([
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
            stand_position_1,
            stand_position_2
        ]) + np.array([front_leg_offset, 0.09, 0])
    elif leg_index == 2:  # Right rear
        waypoints = np.array([
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
            stand_position_1,
            stand_position_2
        ]) + np.array([rear_leg_offset, -0.09, 0])
    else:  # Left rear
        waypoints = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position
        ]) + np.array([rear_leg_offset, 0.09, 0])

    # Calculate which segment we're in
    num_segments = len(waypoints) - 1
    segment_t = t * num_segments % num_segments
    i = int(segment_t)
    segment_phase = segment_t - i

    # Linearly interpolate between waypoints
    return (1 - segment_phase) * waypoints[i] + segment_phase * waypoints[(i + 1) % len(waypoints)]


def cache_target_joint_positions():
    # Calculate and store the target joint positions for a cycle and all 4 legs
    target_joint_positions_cache = []
    target_ee_cache = []
    for leg_index in range(4):
        target_joint_positions_cache.append([])
        target_ee_cache.append([])
        target_joint_positions = [0] * 3
        for t in np.arange(0, 1, 0.02):
            print(t)
            target_ee = interpolate_triangle(t, leg_index)
            target_joint_positions = ik.inverse_kinematics_single_leg(target_ee, leg_index, initial_guess=target_joint_positions)
            target_joint_positions_cache[leg_index].append(target_joint_positions)
    target_joint_positions_cache = np.concatenate(target_joint_positions_cache, axis=1)
    return target_joint_positions_cache

def get_target_joint_position(i, target_joint_positions_cache):
    target_joint_positions = target_joint_positions_cache[i % target_joint_positions_cache.shape[0]]
    return target_joint_positions


def main():
    args = parse_args()
    env = create_env.create_env()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    # initialize the state
    rng = jax.random.PRNGKey(1)

    # grab a trajectory
    state = jit_reset(rng)
    rollout = [state.pipeline_state]
    target_joint_positions_cache = cache_target_joint_positions()
    for i in range(N_STEPS):
        target_joint_position = get_target_joint_position(3*i, target_joint_positions_cache)
        ctrl = jp.array(target_joint_position)
        state = jit_step(state, ctrl)
        
        if args.fix_base:
            qpos = state.pipeline_state.qpos.at[:7].set([0,0,0.5, 1.0, 0,0,0])
            qvel = state.pipeline_state.qvel.at[:6].set([0,0,0, 0,0,0])
            state = state.tree_replace({"pipeline_state.qpos":qpos,"pipeline_state.qvel":qvel})

        rollout.append(state.pipeline_state)

    media.write_video('v.mp4',
        env.render(rollout[::RENDER_EVERY], camera='tracking_cam'),
        fps=1.0 / env.dt / RENDER_EVERY)

if __name__ == "__main__":
    main()

