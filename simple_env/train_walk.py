import argparse
import os
import pickle
import shutil

import wandb
from simple_reward_wrapper import WalkFlat
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def get_train_cfg(args):

    train_cfg_dict = {
        'algorithm': {
            'clip_param': 0.2,
            'desired_kl': 0.01,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'lam': 0.95,
            'learning_rate': 0.001,
            'max_grad_norm': 1.0,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'schedule': 'adaptive',
            'use_clipped_value_loss': True,
            'value_loss_coef': 1.0,
        },
        'init_member_classes': {},
        'policy': {
            'activation': 'elu',
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            'init_noise_std': 1.0,
        },
        'runner': {
            'algorithm_class_name': 'PPO',
            'checkpoint': -1,
            'experiment_name': args.exp_name,
            'load_run': -1,
            'log_interval': 1,
            'max_iterations': args.max_iterations,
            'num_steps_per_env': 24,
            'policy_class_name': 'ActorCritic',
            'record_interval': 100,
            'resume': False,
            'resume_path': None,
            'run_name': '',
            'runner_class_name': 'runner_class_name',
            'save_interval': 100,
        },
        'runner_class_name': 'OnPolicyRunner',
        'seed': 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0, #0
            "FR_hip_joint": 0.0,#1
            "RL_hip_joint": 0.0,#2
            "RR_hip_joint": 0.0,#3
            "FL_thigh_joint": 0.8,#4
            "FR_thigh_joint": 0.8,#5
            "RL_thigh_joint": 1.0,#6
            "RR_thigh_joint": 1.0,#7
            "FL_calf_joint": -1.5,#8
            "FR_calf_joint": -1.5,#9
            "RL_calf_joint": -1.5,#10
            "RR_calf_joint": -1.5,#11
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        'use_terrain': False, # TODO: add terrain later 
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
    }
    obs_cfg = {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.30,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel_x": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_y": -1.0,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_target": 0.5,
        "lin_vel_y_target": 0,
        "ang_vel_target": 0,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()

    gs.init(
        logging_level='warning',
    )

    log_dir = f'logs/{args.exp_name}'
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WalkFlat(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device='cuda:0')


    wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='online')

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True, curriculum=True) # if curriculum is True, it will increase x_target by 0.1 every (max_iter/5) iterations


if __name__ == '__main__':
    main()


'''
# training
python train_walk.py -e go2-walking-v3 --max_iterations 500 

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''