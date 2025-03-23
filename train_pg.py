# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv
from gymnasium.wrappers.time_limit import TimeLimit
from PG import VPG, AdvantageActorCritic
from logger import PGLogger
from reward import RewardTracker
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
            
def main(AGENT_CLASS, dir, checkpoint=None, SAVE_EVERY=100, DEVICE="cuda"):
    # make environment
    env = make_env(max_episodes=4000, gray_scale=True, resize=(128, 128))
    obs, info = env.reset()

    # make agent
    agent = AGENT_CLASS((1, obs.shape[0], obs.shape[1]), grad_acc_batch_size=128)
    agent.train()
    # load saved agent if checkpoint provided
    if checkpoint is not None:
        agent.load(checkpoint)

    # init logger and reward tracker
    logger = PGLogger(f"{dir}/{agent.name}/stats.json")
    rewardTracker = RewardTracker()    

    for episode in range(2000):
        # reset env
        obs, info = env.reset()
        obs = obs[np.newaxis, np.newaxis, :, :].to(DEVICE)

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
            action, log_prob_action, entropy, value_pred = agent.act(obs)
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
            next_obs = next_obs[np.newaxis, np.newaxis, :, :].to(DEVICE)
            obs = next_obs
            info = next_info  
            
            # log step
            logger.log_step(None, reward, next_info['height'], action[0])
            
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
    parser.add_argument("--dir", type=str, default="./pg_results", help="path to store results")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint")
    args = parser.parse_args()
    
    agent_classes = {
        "VPG": VPG,
        "AAC": AdvantageActorCritic
    }
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(agent_classes[args.agent], args.dir, args.checkpoint, DEVICE=DEVICE)