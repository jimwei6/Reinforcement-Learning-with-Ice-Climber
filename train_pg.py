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
            
def main(AGENT_CLASS, dir, checkpoint=None, SAVE_EVERY = 100):
    env = make_env(max_episodes=4000, gray_scale=True, resize=(128, 128))
    obs, info = env.reset()
    agent = AGENT_CLASS((1, obs.shape[0], obs.shape[1]), grad_acc_batch_size=256)
    agent.train()
    logger = PGLogger(f"{dir}/{agent.name}/stats.json")
    rewardTracker = RewardTracker()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load saved agent if checkpoint provided
    if checkpoint is not None:
        agent.load(checkpoint)

    for episode in range(1000):
        obs, info = env.reset()
        rewardTracker.reset()
        ending = ""
        ep_obs = []
        ep_acts = []
        ep_rewards = []
        ep_next_obs = []
        while True:
            # get action and perform
            action, action_num = agent.act(obs[np.newaxis, np.newaxis, :, :])
            next_obs, _, terminated, truncated, next_info  = env.step(action[0])
            done = terminated or truncated

            # Calcluate rewward and store experience
            reward = rewardTracker.calculate_reward(info, next_info, truncated, terminated, action[0])
          
            # append trajectory
            ep_obs.append(obs[np.newaxis, :, :].to(device=device))
            ep_acts.append(action_num)
            ep_rewards.append(reward)
            ep_next_obs.append(next_obs[np.newaxis, :, :].to(device=device))

            # update obs and info
            obs = next_obs
            info = next_info  
            
            # log step
            logger.log_step(None, reward, next_info['height'], action[0])
            
            # Quit episode if game ended         
            if done:
                if truncated:
                    ending = "truncated"
                elif terminated:
                    ending = "gameover" if next_info['lives'] < 0 else "succeed"
                break       

        # Save model every N episodes
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(f"{dir}/{agent.name}/checkpoints/ep_{episode + 1}.chkpt", episode + 1)
        
        # optimize model based on trajectory
        loss = agent.optimize(ep_obs, ep_acts, ep_rewards, ep_next_obs)

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
    
    main(agent_classes[args.agent], args.dir, args.checkpoint)