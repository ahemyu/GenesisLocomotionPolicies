import argparse
import os
import pickle
import torch
from reward_wrapper import FrontFlip
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="frontflip")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("-r", "--record", action="store_true", default=False)
    parser.add_argument("--ckpt", type=int, default=1000)
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    # Load environment configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {"feet_distance": 1}

    env = FrontFlip(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=True,
        debug=True,
    )

    log_dir = f"logs/{args.exp_name}"

    args.max_iterations = 1
    from train_frontflip import get_train_cfg

    runner = OnPolicyRunner(env, get_train_cfg(args), log_dir, device="cuda:0")

    # Load the checkpoint with map_location
    resume_path = os.path.join(log_dir, f'model_{args.ckpt}.pt')
    runner.load(resume_path)
    policy = runner.get_inference_policy(device='cuda:0')

    env.reset()
    obs = env.get_observations()
    with torch.no_grad():
        stop = False
        n_frames = 0
        if args.record:
            env.start_recording(record_internal=False)
        while not stop:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            n_frames += 1
            if args.record:
                if n_frames == 100:
                    env.stop_recording(f"frontflip_{args.ckpt}.mp4")
                    exit()


if __name__ == "__main__":
    main()

"""
# Evaluation command
python eval_frontflip -e experiment_name --ckpt checkpoint_number
"""