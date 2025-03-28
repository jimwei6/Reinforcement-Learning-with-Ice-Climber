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
                 lr = 1e-5,
                 name="VPG",
                 gamma=0.99,
                 beta=0.01,
                 lr_scheduler=False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = self.create_net(input_shape, output_dim).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, gamma=0.5) if lr_scheduler else None
        self.output_dim = output_dim
        self.name = name
        self.lr = lr
        self.training = False
        self.gamma = gamma
        self.beta = beta

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def create_net(self, input_shape, output_dim):
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, output_dim), 
            nn.Softmax(dim=-1)
        )
    
    def act(self, obs):
        obs = obs # (1, C, H, W)
        probs = self.policy(obs)
        action_dist = Categorical(probs)
        action_num = action_dist.sample()
        log_prob_action = action_dist.log_prob(action_num)
        entropy = action_dist.entropy()
        actions = self.convert_nums_to_actions([action_num.item()])
        return actions, log_prob_action, entropy, None, probs.clone().detach().cpu().numpy()

    def convert_nums_to_actions(self, nums):
        res = []
        for num in nums:
            if num == 8:     # 8 means hit
                res.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                res.append([0, 0, 0, 0, 0, 0] + [int(x) for x in format(num, '03b')])
        return res

    def compute_returns(self, rewards):
        discounted_returns = []
        tot_return = 0.0
        
        for rew in reversed(rewards):
            tot_return = rew + self.gamma * tot_return
            discounted_returns.insert(0, tot_return)
        discounted_returns = torch.tensor(discounted_returns, device=self.device, dtype=torch.float32)
        return discounted_returns
    
    def policy_loss(self, log_prob_actions, weights, entropies):
        loss = -(log_prob_actions * weights).sum() - entropies.mean() * self.beta
        return loss
    
    def optimize(self, log_prob_actions, episode_rewards, entropies, values=None, zero_grad=True, batch_size=1, optimizer_step=True):
        if zero_grad:
            self.optimizer.zero_grad()

        returns = self.compute_returns(episode_rewards).detach()
        policy_loss = self.policy_loss(log_prob_actions, returns, entropies) / batch_size
        policy_loss.backward()
        if optimizer_step:
            self.optimize_step()
        return policy_loss.item()

    def optimize_step(self):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
              
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
              'lr': self.lr,
              'gamma': self.gamma,
              'beta': self.beta,
              'scheduler': self.scheduler.state_dict() if self.scheduler is not None else False
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
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']
        self.beta = checkpoint['beta']
        if checkpoint['scheduler'] and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

class ActorCritic(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.policy_net = self.create_policy_net(input_shape, output_dim)
        self.value_net = self.create_value_net(input_shape)

    def create_value_net(self, input_shape):
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8 * 11 * 11, 1), 
            nn.Softmax(dim=-1)
        )
    
    def create_policy_net(self, input_shape, output_dim):
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8 * 11 * 11, output_dim), 
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs):
        action_probs = self.policy_net(obs)
        value = self.value_net(obs)
        return action_probs, value

# Vanilla actor critic
class AdvantageActorCritic(VPG):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 lr = 1e-5,
                 name="AAC",
                 gamma=0.99,
                 beta=0.01,
                 lr_scheduler=False):
        super().__init__(input_shape,
                 output_dim=output_dim,
                 lr = lr,
                 name=name,
                 gamma=gamma,
                 beta=beta,
                 lr_scheduler=lr_scheduler)

    def create_net(self, input_shape, output_dim):
        return ActorCritic(input_shape, output_dim)

    def value_loss(self, values, returns):
        mse = nn.MSELoss()
        return mse(values, returns)
    
    def act(self, obs):
        obs = obs.to(device=self.device) # (1, C, H, W)
        action_probs, value = self.policy(obs)
        action_dist = Categorical(action_probs)
        action_num = action_dist.sample()
        log_prob_action = action_dist.log_prob(action_num)
        entropy = action_dist.entropy()
        actions = self.convert_nums_to_actions([action_num.item()])
        return actions, log_prob_action, entropy, value.flatten(), action_probs.clone().detach().cpu().numpy()
    
    def optimize(self, log_prob_actions, episode_rewards, entropies, values, zero_grad=True, batch_size=1, optimizer_step=True):
        if zero_grad:
            self.optimizer.zero_grad()

        returns = self.compute_returns(episode_rewards)
        policy_loss = self.policy_loss(log_prob_actions, returns, entropies)
        value_loss = self.value_loss(values, returns)
        loss = (policy_loss + 0.5 * value_loss) / batch_size
        loss.backward()
        if optimizer_step:
            self.optimize_step()
        return loss.item()              
    

# Sanity Check
class VPGCartPole(VPG):
    def __init__(self, input_shape, output_dim=2, lr=0.01, name="VPG", gamma=0.99, beta=0.01, lr_scheduler=False):
        super().__init__(input_shape, output_dim, lr, name, gamma, beta, lr_scheduler)
        self.policy.train()

    def create_net(self, input_shape, output_dim):
        return nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def act(self, obs):
        obs = obs.to(device=self.device) # (1, C, H, W)
        probs = self.policy(obs)
        action_dist = Categorical(probs)
        action_num = action_dist.sample()
        log_prob_action = action_dist.log_prob(action_num)
        entropy = action_dist.entropy()
        return action_num.item(), log_prob_action, entropy, None, probs.clone().detach().cpu().numpy()
    