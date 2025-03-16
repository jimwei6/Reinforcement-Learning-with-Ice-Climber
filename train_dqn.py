# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation
from gymnasium.wrappers.time_limit import TimeLimit
from DQN import DQN, PRDQN, DuelingPRDQN, NstepDuelingPRDQN
from logger import DQNLogger
import numpy as np
import argparse

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
    if info is None or len(info) == 0: # initial step 
        next_info['height'] -= 1
        return next_info
    delta = {}
    for k, v in info.items():
        delta[k] = next_info[k] - info[k]
    return delta

def calculate_reward(info, next_info, truncated, terminated, logger):
    delta = calculate_info_delta(info, next_info)
    curr_ep_length = logger.curr_ep_length
    curr_ep_max_height = logger.curr_ep_max_height
    
    if terminated: # rewards at end conditions are large 
        if next_info['lives'] < 0:
            return -100
        elif next_info['height'] >= 10:
            return 100
    elif delta['lives'] < 0:
        return -100
    else:
        rew = 0
        # hitting rewards
        if delta['bricks_hit'] > 0: # a brick was hit
            if next_info['bricks_hit'] < 40: # limit rewards you can get by hitting bricks to 40
                rew += 0.5
        if delta['birds_hit'] > 0: # a bird was hit
            if next_info['birds_hit'] < 10: # limit rewards you can get by hitting birds
                rew += 0.2
        if delta['ice_hit'] > 0: # ice was hit
            if next_info['ice_hit'] < 10: # limit rewards you can get by hitting ice
                rew += 0.2

        # height based rewards
        if delta['height'] > 0: # jumped or moved up
            if next_info['height'] <= curr_ep_max_height:
                rew += 0.9
            else:
                return 10 # going to new height is highly encouraged
        elif delta['height'] < 0:
            rew -= 1
        else: # no change in height
            if next_info['height'] == curr_ep_max_height: # encourages change in height
                rew -= 0.02
            else: # character is below max height
                rew -= 0.1
        return max(-1, min(1, rew)) # keep rewards between [-1, 1]
            
def main(agent_class, dir, checkpoint=None):
    SAVE_EVERY = 100
    env = make_env(max_episodes=4000, gray_scale=True, resize=(128, 128))
    obs, info = env.reset()
    agent = agent_class((1, obs.shape[0], obs.shape[1]), 16)
    if checkpoint is not None:
        agent.load(checkpoint)

    logger = DQNLogger(f"{dir}/{agent.name}/stats.json")
    for episode in range(1000):
        obs, info = env.reset()
        ending = ""
        while True:
            action, action_num = agent.act(obs[np.newaxis, np.newaxis, :, :])
            next_obs, _, terminated, truncated, next_info  = env.step(action[0])
            done = terminated or truncated
            reward = calculate_reward(info, next_info, truncated, terminated, logger)
            agent.cache(obs, next_obs, action, action_num, reward, done)
            obs = next_obs
            info = next_info
            loss = agent.optimize() # can be None if agent does not optimize in this step
            logger.log_step(loss, reward, info['height'], action[0])
            if done:
                if truncated:
                    ending = "truncated"
                elif terminated:
                    ending = "gameover" if info['lives'] < 0 else "succeed"
                break
        if (episode + 1) % SAVE_EVERY == 0:
            agent.save(f"{dir}/{agent.name}/checkpoints/ep_{episode + 1}.chkpt", episode + 1)
        logger.log_episode(info, ending)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["DQN", "PRDQN", "DuelingPRDQN", "NstepDuelingPRDQN"], default="NstepDuelingPRDQN", help="Select the DQN variant")
    parser.add_argument("--dir", type=str, default="./dqn_results", help="path to store results")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint")
    args = parser.parse_args()
    
    agent_classes = {
        "DQN": DQN,
        "PRDQN": PRDQN,
        "DuelingPRDQN": DuelingPRDQN,
        "NstepDuelingPRDQN": NstepDuelingPRDQN,
    }
    
    main(agent_classes[args.agent], args.dir, args.checkpoint)