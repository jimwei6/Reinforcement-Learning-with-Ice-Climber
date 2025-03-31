# Referenced implementations from stable-retro, pytorch mario tutorial
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv, FrameStackMod, CustomRewardWrapper
from gymnasium.wrappers.time_limit import TimeLimit
import gymnasium as gym

from DQN import DQN_CARTPOLE
from logger import PGLogger
from reward import SparseRewardTracker
import numpy as np
import argparse
import torch

def make_env():
    return gym.make("CartPole-v1", max_episode_steps=500)

def main(DEVICE="cuda", LR=1e-2, EPISODES=500, scheduler=False):
    # make environment

    env = make_env()
    obs, info = env.reset()

    # make agent
    agent = DQN_CARTPOLE(obs.shape, lr=LR, replay_size=10000, batch_size=128, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, learn_per_n_steps=1)
    agent.train()
    rew = []
    loss = []
    for episode in range(EPISODES):
        l, r = train(env, agent)
        # if l is not None:
        #     print(l)
        rew.append(r)
        if episode % 10 == 0:
            print(np.mean(rew[episode - 10:episode]), agent.exploration_rate)
            
        
    env.close()

def train(env, policy):
    
    policy.train()
    
    log_prob_actions = []
    rewards = []
    entropies = []
    done = False
    episode_reward = 0

    state, info = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to("cuda")
    while not done:
        action, action_num = policy.act(state) 
        next_state, reward, terminated, truncated, _ = env.step(action_num)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to("cuda")
        policy.cache(state,
                    next_state,
                    torch.tensor(action).to(DEVICE),
                    torch.tensor([action_num]).to(DEVICE),
                    torch.tensor([reward], dtype=torch.float32).to(DEVICE),
                    torch.tensor([done], dtype=torch.float32).to(DEVICE))


        episode_reward += reward
        done = terminated or truncated
        loss = policy.optimize()
        state = next_state

    return loss, episode_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--scheduler', action='store_true', help="Use lr step scheduler")
    parser.add_argument('--episodes', type=int, default=1000, help="episodes to train for")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(DEVICE=DEVICE, LR=args.lr, EPISODES=args.episodes)