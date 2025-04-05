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
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, gamma=0.5) if lr_scheduler else None
        self.output_dim = output_dim
        self.name = name
        self.lr = lr
        self.training = False
        self.gamma = gamma
        self.beta = beta
        self.include_height_info = False

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
    
    def act(self, obs, height=None):
        obs = obs # (1, C, H, W)
        probs = self.policy(obs, height) if self.include_height_info else self.policy(obs)
        action_dist = Categorical(probs)
        action_num = action_dist.sample()
        log_prob_action = action_dist.log_prob(action_num)
        entropy = action_dist.entropy()
        actions = self.convert_nums_to_actions([action_num.item()])
        return actions, log_prob_action, entropy, None, probs.clone().detach().cpu().numpy(), action_num.item()

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
    
    def optimize(self, log_prob_actions, episode_rewards, entropies, values=None, zero_grad=True, batch_size=1, optimizer_step=True, states=None, actions=None):
        if zero_grad:
            self.optimizer.zero_grad()

        returns = self.compute_returns(episode_rewards).detach()
        policy_loss = self.policy_loss(log_prob_actions, returns, entropies) / batch_size
        policy_loss.backward()
        if optimizer_step:
            self.optimize_step()
        return policy_loss.item(), None

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

class VPGHeightInfo(VPG):
    def __init__(self, input_shape, output_dim=9, lr=0.00001, name="VPGHEIGHT", gamma=0.99, beta=0.01, lr_scheduler=False):
        super().__init__(input_shape, output_dim, lr, name, gamma, beta, lr_scheduler)
        self.include_height_info = True

    def create_net(self, input_shape, output_dim):
        return HeightInfoNet(input_shape, output_dim)

class HeightInfoNet(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        c, h, w = input_shape
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.mlp = nn.Sequential(
            nn.Linear(8 * 11 * 11 + 11, output_dim), 
            nn.Softmax(dim=-1)
            )
    
    def forward(self, obs, height):
        conv = self.convnet(obs)
        one_hot = nn.functional.one_hot(height, num_classes=11).unsqueeze(0)
        cat = torch.cat((conv, one_hot), dim=1)
        return self.mlp(cat)

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
    
    def act(self, obs, height=None):
        obs = obs.to(device=self.device) # (1, C, H, W)
        action_probs, value = self.policy(obs, height) if self.include_height_info else self.policy(obs)
        action_dist = Categorical(action_probs)
        action_num = action_dist.sample()
        log_prob_action = action_dist.log_prob(action_num)
        entropy = action_dist.entropy()
        actions = self.convert_nums_to_actions([action_num.item()])
        return actions, log_prob_action, entropy, value.view(-1), action_probs.clone().detach().cpu().numpy(), action_num.item()
    
    def optimize(self, log_prob_actions, episode_rewards, entropies, values, zero_grad=True, batch_size=1, optimizer_step=True, states=None, actions=None):
        if zero_grad:
            self.optimizer.zero_grad()

        returns = self.compute_returns(episode_rewards).detach()
        policy_loss = self.policy_loss(log_prob_actions, returns, entropies) / batch_size
        value_loss = self.value_loss(values, returns) / batch_size
        policy_loss.backward()
        value_loss.backward()

        if optimizer_step:
            self.optimize_step()
        return policy_loss.item(), value_loss.item()

class PPO(AdvantageActorCritic):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 lr = 1e-4,
                 name="PPO",
                 gamma=0.99,
                 beta=0.01,
                 lr_scheduler=False,
                 ppo_clip=0.2,
                 ppo_steps=5):
        super().__init__(input_shape,
                 output_dim=output_dim,
                 lr = lr,
                 name=name,
                 gamma=gamma,
                 beta=beta,
                 lr_scheduler=lr_scheduler)
        self.ppo_clip = ppo_clip
        self.ppo_steps = ppo_steps
    
    def act(self, obs, height=None):
        with torch.no_grad():
          obs = obs.to(device=self.device) # (1, C, H, W)
          action_probs, value = self.policy(obs, height) if self.include_height_info else self.policy(obs)
          action_dist = Categorical(action_probs)
          action_num = action_dist.sample()
          log_prob_action = action_dist.log_prob(action_num)
          entropy = action_dist.entropy()
          actions = self.convert_nums_to_actions([action_num.item()])
        return actions, log_prob_action, entropy, value.view(-1), action_probs.clone().detach().cpu().numpy(), action_num.item()
    
    def compute_advantages(self, returns, values):
        adv = returns - values
        return adv
    
    def policy_loss(self, obj_1, obj_2):
        return torch.min(obj_1, obj_2).sum()

    def optimize(self, log_prob_actions, episode_rewards, entropies, values, actions=None, states=None, zero_grad=None, batch_size=None, optimizer_step=None):
        returns = self.compute_returns(episode_rewards).detach()
        adv = self.compute_advantages(returns, values).detach()
        old_log_probs = log_prob_actions

        total_policy_loss = 0
        total_value_loss = 0
        for _ in range(self.ppo_steps):
            n_action_probs, n_values = self.policy(states)
            action_dist = Categorical(n_action_probs)
            new_log_prob_actions = action_dist.log_prob(actions)
            ratio = (new_log_prob_actions - old_log_probs).exp()

            p_loss_1 = ratio * adv
            p_loss_2 = torch.clamp(ratio, min = 1.0 - self.ppo_clip, max = 1 + self.ppo_clip) * adv   
            policy_loss = self.policy_loss(p_loss_1, p_loss_2)
            value_loss = self.value_loss(n_values.view(-1), returns)
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimize_step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        return total_policy_loss / self.ppo_steps, total_value_loss / self.ppo_steps
    
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
    
    def act(self, obs, height=None):
        obs = obs.to(device=self.device) # (1, C, H, W)
        probs = self.policy(obs, height) if self.include_height_info else self.policy(obs)
        action_dist = Categorical(probs)
        action_num = action_dist.sample()
        log_prob_action = action_dist.log_prob(action_num)
        entropy = action_dist.entropy()
        return action_num.item(), log_prob_action, entropy, None, probs.clone().detach().cpu().numpy()
    