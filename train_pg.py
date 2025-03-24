# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv
from gymnasium.wrappers.time_limit import TimeLimit
from PG import VPG, AdvantageActorCritic
from logger import PGLogger
from reward import SparseRewardTracker, ComplexRewardTracker
import numpy as np
import argparse
import torch

def make_env(max_episodes=None, restricted_actions=retro.Actions.FILTERED, gray_scale=False, resize=None, skip_frames=4):
    env = retro.make('IceClimber-Nes', retro.State.DEFAULT, use_restricted_actions=restricted_actions, players=1)
    env = FrameSkip(env, skip_frames=skip_frames) # 4 steps per action
    env = NormalizeObservation(env, shape=resize)
    env = NoopResetEnv(env)
    if max_episodes is not None: # add max episode length
        env = TimeLimit(env, max_episode_steps=max_episodes)
    if gray_scale:
        env = GrayEnvironment(env)
    return env
            
def main(AGENT_CLASS, REWARD_CLASS, dir, checkpoint=None, SAVE_EVERY=100, DEVICE="cuda", GRAY_SCALE=False, LR=1e-4, EPISODES=1000):
    # make environment
    env = make_env(max_episodes=2500, gray_scale=GRAY_SCALE, resize=(84, 84))
    obs, info = env.reset()
    obs_shape = (1, obs.shape[0], obs.shape[1]) if GRAY_SCALE else obs.shape
    # make agent
    agent = AGENT_CLASS(obs_shape, lr=LR)
    agent.train()
    # load saved agent if checkpoint provided
    if checkpoint is not None:
        agent.load(checkpoint)

    # init logger and reward tracker
    logger = PGLogger(f"{dir}/{agent.name}/stats.json")
    rewardTracker = REWARD_CLASS()    

    for episode in range(EPISODES):
        # reset env
        obs, info = env.reset()
        if GRAY_SCALE:
            obs = obs[np.newaxis, np.newaxis, :, :].to(DEVICE)
        else:
            obs = obs[np.newaxis, :, :].to(DEVICE)

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
            next_obs, _, terminated, truncated, next_info  = env.step(action[0])
            done = terminated or truncated
      
            # Calcluate rewward and store experience
            reward = rewardTracker.calculate_reward(info, next_info, truncated, terminated, action[0])
          
            # append trajectory
            log_prob_actions.append(log_prob_action)
            ep_rewards.append(reward)
            entropies.append(entropy)

            # For Actor critic methods
            if value_pred is not None:
              value_preds.append(value_pred)
    
            # update obs and info
            if GRAY_SCALE:
                next_obs = next_obs[np.newaxis, np.newaxis, :, :].to(DEVICE)
            else:
                next_obs = next_obs[np.newaxis, :, :].to(DEVICE)
            obs = next_obs
            info = next_info  
            
            # log step
            logger.log_step(None, reward, next_info['height'], action[0], probs[0])
            
            # Quit episode if game ended         
            if done or next_info['lives'] < 3:
                if truncated:
                    ending = "truncated"
                else:
                    ending = "gameover" if next_info['lives'] < 3 else "succeed"
                break

        # Save model every N episodes
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(f"{dir}/{agent.name}/checkpoints/ep_{episode + 1}.chkpt", episode + 1)
        
        # optimize model based on trajectory
        loss = agent.optimize(torch.cat(log_prob_actions), ep_rewards, torch.cat(entropies), torch.cat(value_preds) if len(value_preds) else None)

        # log episode
        logger.log_episode(loss, info, ending)
        torch.cuda.empty_cache()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["VPG", "AAC"], default="VPG", help="Select the policy gradient variant")
    parser.add_argument("--reward", type=str, choices=["SPARSE", "COMPLEX"], default="COMPLEX", help="Select the reward variant")
    parser.add_argument("--dir", type=str, default="./pg_results", help="path to store results")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument('--grayscale', action='store_true', help="Convert environment to grayscale")
    parser.add_argument('--episodes', type=int, default=1000, help="episodes to train for")
    args = parser.parse_args()
    
    agent_classes = {
        "VPG": VPG,
        "AAC": AdvantageActorCritic
    }
    reward_classes = {
        "SPARSE": SparseRewardTracker,
        "COMPLEX": ComplexRewardTracker
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(agent_classes[args.agent], reward_classes[args.reward], args.dir, args.checkpoint, DEVICE=DEVICE, GRAY_SCALE=args.grayscale, LR=args.lr, EPISODES=args.episodes)