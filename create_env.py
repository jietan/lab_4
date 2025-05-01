import brax
from brax import envs
from pupperv3_mjx import environment
import numpy as np
import jax
from jax import numpy as jp
from etils import epath
import mediapy as media
from ml_collections import config_dict
from brax.io import mjcf

def get_simulation_config():
    simulation_config = config_dict.ConfigDict()
    simulation_config.model_repo = 'https://github.com/g-levine/pupper_v3_description'
    simulation_config.original_model_path = 'pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
    simulation_config.model_xml = None  # Will be populated when needed
    simulation_config.model_path = "pupper_v3_description/description/mujoco_xml/model_with_obstacles.xml"

    simulation_config.upper_leg_body_names = ["leg_front_r_2", "leg_front_l_2", "leg_back_r_2", "leg_back_l_2"]
    simulation_config.lower_leg_body_names = ["leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3"]
    simulation_config.foot_site_names = [
        "leg_front_r_3_foot_site",
        "leg_front_l_3_foot_site",
        "leg_back_r_3_foot_site",
        "leg_back_l_3_foot_site",
    ]
    simulation_config.foot_radius = 0.02
    simulation_config.torso_name = "base_link"

    simulation_config.max_contact_points = 5
    simulation_config.max_geom_pairs = 4

    sys_temp = mjcf.load(simulation_config.original_model_path)
    joint_upper_limits = sys_temp.jnt_range[1:, 1]
    joint_lower_limits = sys_temp.jnt_range[1:, 0]
    simulation_config.joint_upper_limits = np.array(joint_upper_limits).tolist()
    simulation_config.joint_lower_limits = np.array(joint_lower_limits).tolist()
    
    return simulation_config

def get_reward_config():
    reward_config = config_dict.ConfigDict()
    reward_config.rewards = config_dict.ConfigDict()
    
    scales = config_dict.ConfigDict()
    scales.tracking_lin_vel = 0.0
    scales.tracking_ang_vel = 0.0
    scales.tracking_orientation = 0.0
    scales.lin_vel_z = 0.0
    scales.ang_vel_xy = 0.0
    scales.orientation = 0.0
    scales.torques = -0.0
    scales.joint_acceleration = 0.0
    scales.mechanical_work = 0.0
    scales.action_rate = -0.0
    scales.feet_air_time = 0.0
    scales.stand_still = -0.0
    scales.termination = 0.0
    scales.foot_slip = -0.0
    scales.knee_collision = 0.0
    scales.body_collision = -0.0

    reward_config.rewards.scales = scales
    reward_config.rewards.tracking_sigma = 0.25
    return reward_config


def get_env_config():
    """Creates and returns environment configuration with default settings."""
    
    simulation_config = get_simulation_config()
    reward_config = get_reward_config()
    temp_config = config_dict.ConfigDict()
    temp_config.simulation = simulation_config
    temp_config.reward = reward_config
    CONFIG = config_dict.FrozenConfigDict(temp_config)
    env_kwargs = dict(
        path=CONFIG.simulation.original_model_path,
        observation_history=2,
        action_scale=0.75,
        reward_config=CONFIG.reward,
        foot_site_names=CONFIG.simulation.foot_site_names,
        joint_lower_limits=np.array(CONFIG.simulation.joint_lower_limits),
        joint_upper_limits=np.array(CONFIG.simulation.joint_upper_limits),
        torso_name=CONFIG.simulation.torso_name,
        upper_leg_body_names=CONFIG.simulation.upper_leg_body_names,
        lower_leg_body_names=CONFIG.simulation.lower_leg_body_names,
    )
    return env_kwargs

def create_env(env_name='pupper', **kwargs):
    """Creates and returns a Pupper environment instance.
    
    Args:
        env_name: Name of environment, defaults to 'pupper'
        **kwargs: Optional kwargs to override default env_config
    
    Returns:
        env: Instantiated Brax environment
    """
    env_config = get_env_config()
    if kwargs:
        env_config.update(kwargs)
    envs.register_environment(env_name, environment.PupperV3Env)
    env = envs.get_environment(env_name, **env_config)
    return env
