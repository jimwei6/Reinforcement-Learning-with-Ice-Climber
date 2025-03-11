# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation
from gymnasium.wrappers.time_limit import TimeLimit
from DQN import DQN
from logger import DQNLogger
import numpy as np
import torch

def make_env(max_episodes=None, restricted_actions=retro.Actions.FILTERED, gray_scale=False, resize=None, skip_frames=4):
    env = retro.make('IceClimber-Nes', retro.State.DEFAULT, use_restricted_actions=restricted_actions, players=1)
    env = FrameSkip(env, skip_frames=skip_frames) # 4 steps per action
    env = NormalizeObservation(env, shape=resize)
    if max_episodes is not None: # add max episode length
        env = TimeLimit(env, max_episode_steps=max_episodes)
    if gray_scale:
        env = GrayEnvironment(env)
    return env

def calculate_info_delta(info, next_info):
    if info is None or len(info) == 0:
        next_info['height'] -= 1
        return next_info
    delta = {}
    for k, v in info.items():
        delta[k] = next_info[k] - info[k]
    return delta

def calculate_reward(info, next_info, truncated, terminated):
    delta = calculate_info_delta(info, next_info)
    if truncated:
        return -10 # too slow
    elif terminated:
        if delta['lives'] < 0:
            return -50 # dead
        elif delta['height'] == 10:
            return 50 # completed
    else:
        rew = 0
        if delta['height'] > 0:
            rew += 1
        elif delta['height'] < 0:
            rew -= 2
        else:
            rew -= 0.1

        if delta['bricks_hit'] > 0:
            rew += 0.05
        if delta['birds_hit'] > 0:
            rew += 0.05
        if delta['ice_hit'] > 0:
            rew += 0.05
        return rew
            
def main():
    save_every = 100
    env = make_env(max_episodes=4000, gray_scale=True, resize=(128, 128))
    obs, info = env.reset()
    agent = DQN((1, obs.shape[0], obs.shape[1]), 16)
    logger = DQNLogger('./dqn/stats.json')
    for episode in range(1000):
        obs, info = env.reset()
        while True:
            action, action_num = agent.act(obs[np.newaxis, np.newaxis, :, :])
            next_obs, _, terminated, truncated, next_info  = env.step(action[0])
            done = terminated or truncated
            reward = calculate_reward(info, next_info, truncated, terminated)
            agent.cache(obs[np.newaxis, :, :], next_obs[np.newaxis, :, :], action, action_num, reward, done)
            obs = next_obs
            info = next_info
            loss = agent.optimize()
            if loss is not None: # after buffer warm up starts
                logger.log_step(loss, reward)
            if done:
                break
        if (episode + 1) % save_every == 0:
            agent.save(f"./dqn/checkpoints/ep_{episode + 1}.chkpt")
        logger.log_episode()
    env.close()

if __name__ == "__main__":
    main()
