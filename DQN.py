# Referenced implementations from stable-retro, pytorch mario tutorial
import os
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
import numpy as np

class DQN(nn.Module):
    def __init__(self,
                 input_shape,
                 output_dim,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 0.99999975,
                 batch_size = 64,
                 gamma = 0.99,
                 tau = 0.005,
                 lr = 1e-4,
                 start_learing=256,
                 replay_size=512,
                 loss_fn=nn.SmoothL1Loss):
        super(DQN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.online_net = self.create_net(input_shape, output_dim).to(device=self.device)
        self.target_net = self.create_net(input_shape, output_dim).to(device=self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.memory = TensorDictReplayBuffer(storage=LazyTensorStorage(replay_size, device=torch.device("cuda")))
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr, amsgrad=True)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.exploration_rate = eps_start
        self.exploration_rate_decay = eps_decay
        self.exploration_rate_min = eps_end
        self.steps = 0
        self.start_learning = max(start_learing, batch_size)
        self.loss_fn = loss_fn()
    
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
        )
    
    def recall(self, batch_size):
        batch = self.memory.sample(batch_size)
        state, next_state, action, action_num, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "action_num", "reward", "done"))
        return state, next_state, action, action_num, reward.squeeze(), done.squeeze()
    
    def cache(self, state, next_state, action, action_num, reward, done):
        action = torch.tensor(action)
        action_num = torch.tensor([action_num])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "action_num": action_num, "reward": reward, "done": done}, batch_size=[]))
    
    def act(self, obs): 
        # select based on epsilon greedy policy (trade off between exploitation and exploration)
        if np.random.rand() < self.exploration_rate: # explore
            action_num = np.random.randint(self.output_dim)
            actions = self.convert_nums_to_actions([action_num])
        else: # exploit
            with torch.no_grad():
                obs = obs.to(device=self.device) # (1, C, H, W)
                action_values = self.online_net(obs)
                action_num = torch.argmax(action_values, axis=1).item()
                actions = self.convert_nums_to_actions([action_num])

        self.exploration_rate *= self.exploration_rate_decay # decay exploration rate
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.steps += 1

        return actions, action_num

    def convert_nums_to_actions(self, nums):
        return [[0, 0, 0, 0, 0] + [(num >> i) & 1 for i in range(3, -1, -1)] for num in nums]

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        if len(self.memory) < self.start_learning:
            return
 
        state, next_state, action, action_num, reward, done = self.recall(self.batch_size)
        td_estimate = self.online_net(state).gather(1, action_num)

        with torch.no_grad():
            online_next_actions = torch.argmax(self.online_net(next_state), axis=1)
            target_next_values = self.target_net(next_state).gather(1, online_next_actions.unsqueeze(1))
            td_target = reward.unsqueeze(1) + (1 - done.unsqueeze(1).float()) * self.gamma * target_next_values
        loss = self.loss_fn(td_estimate, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft updates
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.online_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        return loss.item()

    def save(self, path):
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        torch.save(
            dict(online=self.online_net.state_dict(), targe=self.target_net.state_dict(), exploration_rate=self.exploration_rate),
            path
        )