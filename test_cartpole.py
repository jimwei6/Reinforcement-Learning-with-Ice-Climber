# Referenced implementations from stable-retro, pytorch mario tutorial
from env_utils import FrameSkip, GrayEnvironment, NormalizeObservation, NoopResetEnv, FrameStackMod, CustomRewardWrapper
from gymnasium.wrappers.time_limit import TimeLimit
import gymnasium as gym

from PG import VPG, AdvantageActorCritic, VPGCartPole
from logger import PGLogger
from reward import SparseRewardTracker, ComplexRewardTracker
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
    agent = VPGCartPole(obs.shape, lr=LR)
    agent.train()
    rew = []
    for episode in range(EPISODES):
        rew.append(train(env, agent)[1])
        if episode % 10 == 0:
            print(np.mean(rew[episode - 10:episode]))
        
    env.close()

def train(env, policy):
    
    policy.train()
    
    log_prob_actions = []
    rewards = []
    entropies = []
    done = False
    episode_reward = 0

    state, info = env.reset()
    
    while not done:

        state = torch.FloatTensor(state).unsqueeze(0).to("cuda")
        action, log_prob_action, entropy, _, _ = policy.act(state)    
        state, reward, terminated, truncated, _ = env.step(action)


        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        entropies.append(entropy)

        episode_reward += reward
        done = terminated or truncated
    log_prob_actions = torch.cat(log_prob_actions)
    loss = policy.optimize(log_prob_actions, rewards, torch.cat(entropies))    
    return loss, episode_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--scheduler', action='store_true', help="Use lr step scheduler")
    parser.add_argument('--episodes', type=int, default=1000, help="episodes to train for")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    main(DEVICE=DEVICE, LR=args.lr, EPISODES=args.episodes)