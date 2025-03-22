import os
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from torch.distributions.categorical import Categorical
from tensordict import TensorDict
import numpy as np
from collections import deque

# Vanilla Policy Gradient (REINFORCE). Reference implementation https://spinningup.openai.com/en/latest/algorithms/vpg.html
class VPG(nn.Module):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 lr = 1e-4,
                 name="VPG",
                 grad_acc_batch_size=None,
                 gamma=0.99):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = self.create_net(input_shape, output_dim).to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.output_dim = output_dim
        self.name = name
        self.grad_acc_batch_size = grad_acc_batch_size
        self.lr = lr
        self.training = False
        self.gamma = gamma

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def create_net(self, input_shape, output_dim):
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, output_dim),
            nn.Softmax(dim=1)
        )
    
    def act(self, obs):
        with torch.no_grad():
            obs = obs.to(device=self.device) # (1, C, H, W)
            action_dist = Categorical(probs=self.policy(obs))
            action_num = action_dist.sample()
            actions = self.convert_nums_to_actions([action_num.item()])
        return actions, action_num

    def convert_nums_to_actions(self, nums):
        res = []
        for num in nums:
            if num == 8:     # 8 means hit
                res.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                res.append([0, 0, 0, 0, 0, 0] + [int(x) for x in format(num, '03b')])
        return res

    def compute_returns(self, rewards):
        discounted_returns = torch.zeros(len(rewards)).to(device=self.device)
        tot_return = 0.0
        
        for t in reversed(range(len(rewards))):
            tot_return = rewards[t] + self.gamma * tot_return
            discounted_returns[t] = tot_return
            
        return discounted_returns
    
    def optimize_policy_batch(self, obs_tensor, actions_tensor, returns):
        self.optimizer.zero_grad()
        loss = self.compute_loss(obs_tensor, actions_tensor, returns)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def optimize_policy(self, obs_tensor, actions_tensor, returns):
        if self.grad_acc_batch_size is None:
            return self.optimize_policy_batch(obs_tensor, actions_tensor, returns)
        else: # batched optimization to deal with memory size issues
            losses = []
            num_batches = len(obs_tensor) // self.grad_acc_batch_size
            if len(obs_tensor) % self.grad_acc_batch_size != 0:
                num_batches += 1
            for i in range(num_batches):
                start = i * self.grad_acc_batch_size
                end = min(start + self.grad_acc_batch_size, len(obs_tensor))
                obs_batch = obs_tensor[start:end]
                actions_batch = actions_tensor[start:end]
                rewards_batch = returns[start:end]
                loss = self.optimize_policy_batch(obs_batch, actions_batch, rewards_batch)
                losses.append(loss)
            return np.mean(losses)
    
    def optimize(self, episode_obs, episode_actions, episode_rewards, episode_next_obs=None):
        returns = self.compute_returns(episode_rewards)
        obs_tensor = torch.stack(episode_obs)
        actions_tensor = torch.tensor(episode_actions, device=self.device)
        return self.optimize_policy(obs_tensor, actions_tensor, returns)
    
    def compute_loss(self, obs, action, weights):
        logp = Categorical(probs=self.policy(obs)).log_prob(action)
        return -(logp * weights).mean()
              
    def save(self, path, episode):
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        torch.save(self.make_save_obj(episode), path)
        
    def make_save_obj(self, episode):
        return {
              'policy': self.policy.state_dict(),
              'output_dim': self.output_dim,
              'name': self.name,
              'optimizer': self.optimizer.state_dict(),
              'episode': episode,
              'grad_acc_batch_size': self.grad_acc_batch_size,
              'lr': self.lr,
              'gamma': self.gamma
        }

    def load(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.load_checkpoint(checkpoint)
            print(f"Model loaded successfully")
        else:
            print(f"Error: The file {path} does not exist.")

    def load_checkpoint(self, checkpoint):
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.output_dim = checkpoint['output_dim']
        self.name = checkpoint['name']
        self.grad_acc_batch_size = checkpoint['grad_acc_batch_size']
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']

# Vanilla actor critic
class AdvantageActorCritic(VPG):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 lr = 1e-4,
                 name="AAC",
                 gamma=0.99,
                 tau=0.95,
                 grad_acc_batch_size=None):
        super().__init__(input_shape,
                 output_dim,
                 lr,
                 name,
                 grad_acc_batch_size)
        self.value_network = self.create_value_network(input_shape).to(device=self.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau

    def create_value_network(self, input_shape):
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, 1),
        )
    
    def make_save_obj(self, episode):
        obj = super().make_save_obj(episode)
        obj['value'] = self.value_network.state_dict()
        obj['value_optimizer'] = self.value_optimizer.state_dict()
        return obj
    
    def load_checkpoint(self, checkpoint):
        self.value_network.load_state_dict(checkpoint['value'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

    def optimize_value_batch(self, values, returns):
        self.value_optimizer.zero_grad()
        mse = nn.MSELoss()
        loss = mse(values, returns)
        loss.backward()
        self.value_optimizer.step()
        return loss.item()
    
    def optimize_value(self, values, returns):
        if self.grad_acc_batch_size is None:
            return self.optimize_value_batch(values, returns)
        else:
            losses = []
            num_batches = len(values) // self.grad_acc_batch_size
            if len(values) % self.grad_acc_batch_size != 0:
                num_batches += 1

            for i in range(num_batches):
                start = i * self.grad_acc_batch_size
                end = min(start + self.grad_acc_batch_size, len(values))
                values_batch = values[start:end]
                returns_batch = returns[start:end]
                loss = self.optimize_value_batch(values_batch, returns_batch)
                losses.append(loss)
            return np.mean(losses)

    def optimize(self, episode_obs, episode_actions, episode_rewards, episode_next_obs):
        returns = self.compute_returns(episode_rewards).detach()
        values = self.value_network(torch.stack(episode_obs)).squeeze()
        value_loss = self.optimize_value(values, returns)

        advantages = returns - values.detach()
        obs_tensor = torch.stack(episode_obs)
        actions_tensor = torch.tensor(episode_actions, device=self.device)
        policy_loss = self.optimize_policy(obs_tensor, actions_tensor, advantages)
        return [policy_loss, value_loss]