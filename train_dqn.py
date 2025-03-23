# Referenced implementations from stable-retro, pytorch mario tutorial
import retro
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv
from gymnasium.wrappers.time_limit import TimeLimit
from DQN import DQN, PRDQN, DuelingPRDQN, NstepDuelingPRDQN
from logger import DQNLogger
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
            
def main(agent_class, dir, checkpoint=None, SAVE_EVERY=100, DEVICE="cuda"):
    # make env
    env = make_env(max_episodes=3000, gray_scale=True, resize=(128, 128))
    obs, info = env.reset()

    # make agent
    agent = agent_class((1, obs.shape[0], obs.shape[1]))
    agent.train()
    if checkpoint is not None:
        agent.load(checkpoint)

    # init logger and reward tracker
    logger = DQNLogger(f"{dir}/{agent.name}/stats.json")
    rewardTracker = RewardTracker()

    for episode in range(2000):
        # reset env
        obs, info = env.reset()
        obs = obs[np.newaxis, np.newaxis, :, :].to(DEVICE)

        # reset rewards
        rewardTracker.reset()
        
        # reset episode variables
        ending = ""

        # reset agent exploration rate (decaying) to allow exploration across episodes
        agent.reset_episode(0.95 * np.exp(-episode / 10))

        while True:
            # get action and perform
            action, action_num = agent.act(obs)
            next_obs, _, terminated, truncated, next_info  = env.step(action[0])
            done = terminated or truncated
            next_obs = next_obs[np.newaxis, np.newaxis, :, :].to(DEVICE)

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
            logger.log_step(loss, reward, next_info['height'], action[0])

            # Quit episode if game ended         
            if done or next_info['lives'] < 3:
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
    args = parser.parse_args()
    
    agent_classes = {
        "DQN": DQN,
        "PRDQN": PRDQN,
        "DuelingPRDQN": DuelingPRDQN,
        "NstepDuelingPRDQN": NstepDuelingPRDQN,
    }
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(agent_classes[args.agent], args.dir, args.checkpoint, DEVICE=DEVICE)