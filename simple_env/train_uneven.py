import argparse
import json
import os
import pickle
import shutil

import wandb
from simple_reward_wrapper import WalkUneven
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

 
def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2, # control how much the policy is allowed to change at each update step (increase => faster learning but riskier, decrease => slower but more stable)
            "desired_kl": 0.01, # how much change do I want between the old and new policy (using an adaptive schedule in this implementation)
            "entropy_coef": 0.01, # rewards randomness in action selection (might make sense to set it higher in early training and lower it later)
            "gamma": 0.99, # determines how much agent values future rewards 
            "lam": 0.95, # lambda parameter for GAE (Generalized Advantage Estimation); higher means advantages depend more on long-term returns, lower means more on short-term returns
            "learning_rate": 0.001, # is adaptive 
            "max_grad_norm": 1.0, # gradient clipping (to prevent exploding gradients)
            "num_learning_epochs": 5, # how often do we reuse one rollout batch 
            "num_mini_batches": 4, # how many chunks the data is split into during training ((num_envs * num_steps_per_env) / num_mini_batches) 
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0, # weight of value loss in the total loss function; so 1.0 means value loss is equally important as policy loss
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 128, #length of trajectories
            "policy_class_name": "ActorCritic",
            "record_interval": 100,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 200,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
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
        "kp": 20.0, # we could try lowering to 15 or 10
        "kd": 0.5, # if joint movements feel springy/jittery we could try increasing to 1.0
        # termination
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 30.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.30,
        "simulate_action_latency": True,
        "clip_actions": 100.0, # self.actions = torch.clip(actions, -clip_actions, clip_actions), so it prevents the actions from going outside the range of -100 to 100 (which is too high)
        'use_terrain': True,
        'terrain_cfg': {
            'subterrain_types': "pyramid_stairs_terrain",
            'n_subterrains': (3, 3),
            'subterrain_size': (12.0, 12.0),
            'horizontal_scale': 0.25,
            'vertical_scale': 0.005,
        },
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
    }

    obs_cfg = {
        "num_obs": 45,
        "num_priviliged_obs": 75,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.40,         # controls how quickly the reward falls off with increasing error
        "lin_vel_target": 1.0,          # target linear velocity
        "reward_scales": {
            # "lin_vel_x": 1.0,            # Reward for x axis base linear velocity
            "tracking_lin_vel_x": 1.0,    # Reward for tracking x axis base linear velocity
            "forward_progress": 1.0,        # Reward for forward progress
            "lin_vel_y": -5.,           # Penalty for y axis base linear velocity
            "lin_vel_z": -0.1,           # Penalty for z axis base linear velocity
            "action_rate": -0.005,           # Small penalty for rapid action changes
            "orientation": -0.01,          # Penalty for orientation not parallel to terrain
            "collision": -1.0,            # Penalty for collision
        },
    }
    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-uneven")
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=1)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg= get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WalkUneven(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    all_cfgs = {
        "env_cfg": env_cfg,
        "obs_cfg": obs_cfg,
        "reward_cfg": reward_cfg,
        "train_cfg": train_cfg,
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(all_cfgs, f, indent=4)

    wandb.init(project='genesis', name=args.exp_name, dir=log_dir, mode='online')
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
To only see one of the GPUs: export CUDA_VISIBLE_DEVICES=1 (or 0)
python train_uneven.py -e go2-uneven-v1 -B 8192 --max_iterations 1000
"""