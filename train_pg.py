# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv, FrameStackMod, CustomRewardWrapper
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers import RecordVideo

from PG import VPG, AdvantageActorCritic
from logger import PGLogger
from reward import SparseRewardTracker
import numpy as np
import argparse
import torch
import random

# Seed for now
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def make_env(rewardTracker, max_episodes=None, restricted_actions=retro.Actions.FILTERED, gray_scale=False, resize=None,
              skip_frames=4, video_episode=None, video_dir=None, render_mode="rgb_array"):
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
            
def main(AGENT_CLASS, REWARD_CLASS, DIR, CHECKPOINT=None, SAVE_EVERY=50,
          DEVICE="cuda", RGB=False, LR=1e-4, EPISODES=1000, scheduler=False,
          TRAJ_BATCH_SIZE=8, VIDEO_EPISODE=None, VIDEO_DIR="./videos", RENDER_MODE="rgb_array"):
    rewardTracker = REWARD_CLASS()    
    # make environment

    env = make_env(rewardTracker, max_episodes=2500, gray_scale=not RGB, resize=(88, 88), video_episode=VIDEO_EPISODE, video_dir=VIDEO_DIR, render_mode=RENDER_MODE)
    obs, info = env.reset(seed=seed)

    # make agent
    agent = AGENT_CLASS(obs.shape, lr=LR, lr_scheduler=scheduler)
    agent.train()
    # load saved agent if checkpoint provided
    if CHECKPOINT is not None:
        agent.load(CHECKPOINT)

    # init logger and reward tracker
    logger = PGLogger(f"{DIR}/{agent.name}/stats.json")

    for episode in range(EPISODES):
        batch_loss = 0
        for batch in range(TRAJ_BATCH_SIZE):
            # reset env
            obs, info = env.reset()
            obs = obs.unsqueeze(0).to(DEVICE)
            # reset rewards
            rewardTracker.reset()

            # reset episode variables
            ending = ""
            log_prob_actions = []
            ep_rewards = []
            entropies = []
            value_preds = []
            
            while True:
                # get action and perform
                action, log_prob_action, entropy, value_pred, probs = agent.act(obs)
                next_obs, reward, terminated, truncated, next_info  = env.step(action[0], info)
                done = terminated or truncated
                info = next_info  

                # append trajectory
                log_prob_actions.append(log_prob_action)
                ep_rewards.append(reward)
                entropies.append(entropy)

                # For Actor critic methods
                if value_pred is not None:
                  value_preds.append(value_pred)
        
                # update obs and info
                next_obs = next_obs.unsqueeze(0).to(DEVICE)
                obs = next_obs
                
                # log step
                logger.log_step(None, reward, next_info['height'], action[0], probs[0])
                
                # Quit episode if game ended         
                if done:
                    if truncated:
                        ending = "truncated"
                    else:
                        ending = "gameover" if next_info['lives'] < 3 else "succeed"
                    break
            # log episode
            logger.log_episode(ending)
            batch_loss += agent.optimize(torch.cat(log_prob_actions),
                                        ep_rewards,
                                        torch.cat(entropies),
                                        torch.cat(value_preds) if len(value_preds) else None,
                                        zero_grad=batch==0,
                                        batch_size=TRAJ_BATCH_SIZE,
                                        optimizer_step=False) # accumulate gradient across batch
        # optimize model on accumulated gradients
        agent.optimize_step()
        logger.log_batch(batch_loss)
        torch.cuda.empty_cache()
        # Save model every N episodes
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(f"{DIR}/{agent.name}/checkpoints/batch_{episode + 1}.chkpt", episode + 1)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["VPG", "AAC"], default="VPG", help="Select the policy gradient variant")
    parser.add_argument("--reward", type=str, choices=["SPARSE"], default="SPARSE", help="Select the reward variant")
    parser.add_argument("--dir", type=str, default="./pg_results", help="path to store results")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--rgb', action='store_true', help="Convert environment to rgb")
    parser.add_argument('--scheduler', action='store_true', help="Use lr step scheduler")
    parser.add_argument('--episodes', type=int, default=1000, help="episodes to train for")
    parser.add_argument('--grad-acc-size', type=int, default=1, help="Number of trajectories to accumulate gradient before optimizing model")
    parser.add_argument('--save-video-episode', type=int, default=None, help="Save video of runs per episode specified")
    parser.add_argument('--render', action='store_true', help="Render environment")
    args = parser.parse_args()
    
    agent_classes = {
        "VPG": VPG,
        "AAC": AdvantageActorCritic
    }
    reward_classes = {
        "SPARSE": SparseRewardTracker
    }
    RENDER_MODE = "human" if args.render else "rgb_array"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(agent_classes[args.agent], reward_classes[args.reward], args.dir, args.checkpoint,
          DEVICE=DEVICE, RGB=args.rgb, LR=args.lr, EPISODES=args.episodes, TRAJ_BATCH_SIZE=args.grad_acc_size,
          VIDEO_EPISODE=args.save_video_episode, VIDEO_DIR=f"{args.dir}/{args.agent}/videos", RENDER_MODE=RENDER_MODE)