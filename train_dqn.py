# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv, FrameStackMod, CustomRewardWrapper
from gymnasium.wrappers.time_limit import TimeLimit
from DQN import DQN, PRDQN, DuelingPRDQN, NstepDuelingPRDQN
from gymnasium.wrappers import RecordVideo
from logger import DQNLogger
from reward import SparseRewardTracker, DQNRewardTracker
import numpy as np
import argparse
import torch
import random


def make_env(rewardTracker, max_episodes=None, restricted_actions=retro.Actions.FILTERED, gray_scale=False,
              resize=None, skip_frames=4, video_episode=None, video_dir=None, render_mode="rgb_array"):
    env = retro.make('IceClimber-Nes', retro.State.DEFAULT, use_restricted_actions=restricted_actions, players=1, render_mode=render_mode)
    env = NormalizeObservation(env, shape=resize)
    env = NoopResetEnv(env)
    if max_episodes is not None: # add max episode length
        env = TimeLimit(env, max_episode_steps=max_episodes * skip_frames)
    if gray_scale:
        env = GrayEnvironment(env)
    if video_episode is not None:
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: x % video_episode == 0, name_prefix="climber")
    env = FrameStackMod(env, 4) # stack frames for temporal information
    env = CustomRewardWrapper(env, rewardTracker)
    env = FrameSkip(env, skip_frames=skip_frames) # 4 steps per action
    return env
            
def main(agent_class, REWARD_CLASS, dir, checkpoint=None, SAVE_EVERY=100, DEVICE="cuda", RGB=False, LR=1e-4,
         EPISODES=1000, REPLAY_SIZE=512, BATCH_SIZE=64, EPS_DECAY=1000, BUFFER_DEVICE="cpu", 
         VIDEO_EPISODE=None, VIDEO_DIR="./videos", RENDER_MODE="rgb_array", BUFFER_DIR="./buffer",
         START_LEARNING=128):
    rewardTracker = REWARD_CLASS()
    # make env
    env = make_env(rewardTracker, max_episodes=2500, gray_scale=not RGB,
                    resize=(88, 88), video_episode=VIDEO_EPISODE, video_dir=VIDEO_DIR, render_mode=RENDER_MODE)
    obs, info = env.reset()
    # make agent
    agent = agent_class(obs.shape, lr=LR, replay_size=REPLAY_SIZE, batch_size=BATCH_SIZE,
                         eps_decay=EPS_DECAY, buffer_device=BUFFER_DEVICE, device=DEVICE, buffer_dir=BUFFER_DIR,
                         start_learning=START_LEARNING)
    agent.train()
    if checkpoint is not None:
        agent.load(checkpoint)

    # init logger and reward tracker
    logger = DQNLogger(f"{dir}/{agent.name}/stats.json")

    for episode in range(EPISODES):
        # reset env
        obs, info = env.reset()
        obs = obs.unsqueeze(0).to(DEVICE)

        # reset rewards
        rewardTracker.reset()
        
        # reset episode variables
        ending = ""

        # reset agent exploration rate (decaying) to allow exploration across episodes since we have small buffer
        # agent.reset_episode(0.95 * np.exp(-episode / 10))

        while True:
            # get action and perform
            action, action_num = agent.act(obs)
            next_obs, _, terminated, truncated, next_info  = env.step(action[0], info)
            done = terminated or truncated

            next_obs = next_obs.unsqueeze(0).to(DEVICE)

            # Calcluate rewward and store experience
            reward = rewardTracker.calculate_reward(info, next_info, truncated, terminated, action[0])
            agent.cache(obs,
                        next_obs,
                        torch.tensor(action).to(DEVICE),
                        torch.tensor([action_num]).to(DEVICE),
                        torch.tensor([reward], dtype=torch.float32).to(DEVICE),
                        torch.tensor([done], dtype=torch.float32).to(DEVICE))
            
            # optimize dqn; can be None if agent is not set to optimize during the step
            loss = agent.optimize()

            # log step
            logger.log_step(loss, reward, next_info['height'], action[0], agent.exploration_rate)

            # Quit episode if game ended         
            if done:
                if truncated:
                    ending = "truncated"
                else:
                    ending = "gameover" if next_info['lives'] < 3 else "succeed"
                break
            
            # update observations and current info
            obs = next_obs
            info = next_info

        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(f"{dir}/{agent.name}/checkpoints/ep_{episode + 1}.chkpt", episode + 1)
        
        logger.log_episode(info, ending)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["DQN", "PRDQN", "DuelingPRDQN", "NstepDuelingPRDQN"], default="NstepDuelingPRDQN", help="Select the DQN variant")
    parser.add_argument("--dir", type=str, default="./dqn_results", help="path to store results")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--reward", type=str, choices=["DQN", "SPARSE"], default="DQN", help="Select the reward variant")
    parser.add_argument('--rgb', action='store_true', help="Convert environment to grayscale")
    parser.add_argument('--episodes', type=int, default=1000, help="episodes to train for")
    parser.add_argument("--memory", type=int, default=512, help="Replay Memory Size")
    parser.add_argument("--batch", type=int, default=64, help="Batch sample size for each optimization step")
    parser.add_argument("--decay", type=float, default=1000, help="Number of steps for exloration to decay to min rate")
    parser.add_argument("--buffer-device", choices=["cpu", "cuda"], default='cpu', help="Device to store buffer on")
    parser.add_argument('--save-video-episode', type=int, default=None, help="Save video of runs per episode specified")
    parser.add_argument('--render', action='store_true', help="Render environment")
    parser.add_argument('--start-learning', type=int, default=128, help="Number of experiences in buffer before starting to learn")
    args = parser.parse_args()
    
    agent_classes = {
        "DQN": DQN,
        "PRDQN": PRDQN,
        "DuelingPRDQN": DuelingPRDQN,
        "NstepDuelingPRDQN": NstepDuelingPRDQN,
    }

    reward_classes = {
        "SPARSE": SparseRewardTracker,
        "DQN": DQNRewardTracker
    }
    RENDER_MODE = "human" if args.render else "rgb_array"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(agent_classes[args.agent], reward_classes[args.reward], args.dir, args.checkpoint,
          DEVICE=DEVICE, RGB=args.rgb, LR=args.lr, EPISODES=args.episodes,
          BATCH_SIZE=args.batch, REPLAY_SIZE=args.memory, EPS_DECAY=args.decay, BUFFER_DEVICE=args.buffer_device,
          VIDEO_EPISODE=args.save_video_episode, VIDEO_DIR=f"{args.dir}/{args.agent}/videos/", RENDER_MODE=RENDER_MODE,
          BUFFER_DIR=f"{args.dir}/{args.agent}/buffer/",
          START_LEARNING=args.start_learning)