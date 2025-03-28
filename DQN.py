# Referenced implementations from stable-retro, pytorch mario tutorial
import os
import torch
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tensordict import TensorDict
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 eps_start = 0.95,
                 eps_end = 0.05,
                 eps_decay = 1000,
                 batch_size = 64,
                 gamma = 0.95,
                 tau = 0.005,
                 lr = 1e-4,
                 start_learning=128,
                 replay_size=512,
                 loss_fn=nn.MSELoss,
                 name="DQN",
                 learn_per_n_steps=1,
                 buffer_device="cpu",
                 device="cpu"):
        super().__init__()
        self.device = device
        self.online_net = self.create_net(input_shape, output_dim).to(device=self.device)
        self.target_net = self.create_net(input_shape, output_dim).to(device=self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.memory = self.create_memory_buffer(replay_size=replay_size, device=buffer_device)
        self.replay_size = replay_size
        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr, amsgrad=True)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.exploration_rate = eps_start
        self.exploration_rate_decay = eps_decay
        self.exploration_rate_max = eps_start
        self.exploration_rate_min = eps_end
        self.steps = 0
        self.start_learning = max(start_learning, batch_size)
        self.loss_fn = loss_fn()
        self.name = name
        self.learn_per_n_steps = learn_per_n_steps
        self.lr = lr
        self.training = False
        self.buffer_device = buffer_device
  
    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def create_memory_buffer(self, replay_size, device):
        return TensorDictReplayBuffer(storage=LazyTensorStorage(replay_size, device=device))

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
    
    def recall(self, batch_size):
        batch = self.memory.sample(batch_size)
        obs, next_obs, action, action_num, reward, done = (batch.get(key) for key in ("obs", "next_obs", "action", "action_num", "reward", "done"))
        if self.buffer_device == self.device:
            return obs.squeeze(1), next_obs.squeeze(1), action, action_num, reward, done
        return obs.squeeze(1).to(self.device), next_obs.squeeze(1).to(self.device), action.to(self.device), action_num.to(self.device), reward.to(self.device), done.to(self.device)    
    
    def cache(self, obs, next_obs, action, action_num, reward, done):
        if self.device != self.buffer_device:
            self.memory.add(TensorDict({"obs": obs.to(self.buffer_device), "next_obs": next_obs.to(self.buffer_device),
                                        "action": action.to(self.buffer_device), "action_num": action_num.to(self.buffer_device), 
                                        "reward": reward.to(self.buffer_device), "done": done.to(self.buffer_device)}, batch_size=[]))
        else:
            self.memory.add(TensorDict({"obs": obs, "next_obs": next_obs, "action": action, "action_num": action_num, "reward": reward, "done": done}, batch_size=[]))

    def act(self, obs): 
        # select based on epsilon greedy policy (trade off between exploitation and exploration)
        if self.training and np.random.rand() < self.exploration_rate: # explore
            action_num = np.random.randint(self.output_dim)
            actions = self.convert_nums_to_actions([action_num])
        else: # exploit
            with torch.no_grad():
                obs = obs.to(device=self.device) # (1, C, H, W)
                action_values = self.online_net(obs)
                action_num = torch.argmax(action_values, axis=1).item()
                actions = self.convert_nums_to_actions([action_num])

        if self.training:
            self.update_exploration_rate()
            self.steps += 1
        return actions, action_num
    
    def update_exploration_rate(self):
        self.exploration_rate = self.exploration_rate_max * np.exp(-1. * self.steps / self.exploration_rate_decay) # decay exploration rate slowly across ~ 5 episodes
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    def reset_episode(self, new_max_exploration_rate=None):
        if new_max_exploration_rate is not None:
            self.exploration_rate_max = new_max_exploration_rate
        self.exploration_rate = self.exploration_rate_max
        self.steps = 0

    def convert_nums_to_actions(self, nums):
        res = []
        for num in nums:
            if num == 8:     # 8 means hit
                res.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                res.append([0, 0, 0, 0, 0, 0] + [int(x) for x in format(num, '03b')])
        return res
    
    def soft_update_target(self):
        # Soft updates
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.online_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def update_online(self):
        obs, next_obs, action, action_num, reward, done = self.recall(self.batch_size)
        td_estimate = self.online_net(obs).gather(1, action_num)
        with torch.no_grad():
            online_next_actions = torch.argmax(self.online_net(next_obs), axis=1)
            target_next_values = self.target_net(next_obs).gather(1, online_next_actions.unsqueeze(1))
        
        td_target = reward + (1 - done) * self.gamma * target_next_values
        loss = self.loss_fn(td_estimate, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        if len(self.memory) < self.start_learning:
            return
        if self.steps % self.learn_per_n_steps == 0:
            loss = self.update_online()
            self.soft_update_target()
            return loss
        return None

    def save(self, path, episode):
        log_dir = os.path.dirname(path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        torch.save(self.make_save_obj(episode),
            path)
        
    def make_save_obj(self, episode):
        return {
              'online': self.online_net.state_dict(), 
              'target': self.target_net.state_dict(),
              'exploration_rate': self.exploration_rate,
              'output_dim': self.output_dim,
              'batch_size': self.batch_size,
              'gamma': self.gamma,
              'tau': self.tau,
              'exploration_rate_decay': self.exploration_rate_decay,
              'exploration_rate_min': self.exploration_rate_min,
              'exploration_rate_max': self.exploration_rate_max,
              'exploration_rate': self.exploration_rate,
              'steps': self.steps,
              'start_learning': self.start_learning,
              'loss_fn': self.loss_fn.__class__.__name__,
              'name': self.name,
              'learn_per_n_steps': self.learn_per_n_steps,
              'optimizer': self.optimizer.state_dict(),
              'episode': episode,
              'lr': self.lr,
              'replay_size': self.replay_size
        }

    def load(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.load_checkpoint(checkpoint)
            print(f"Model loaded successfully")
        else:
            print(f"Error: The file {path} does not exist.")

    def load_checkpoint(self, checkpoint):
        self.online_net.load_state_dict(checkpoint['online'])
        self.target_net.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.output_dim = checkpoint['output_dim']
        self.batch_size = checkpoint['batch_size']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.exploration_rate_decay = checkpoint['exploration_rate_decay']
        self.exploration_rate_min = checkpoint['exploration_rate_min']
        self.exploration_rate_max = checkpoint['exploration_rate_max']
        self.exploration_rate = checkpoint['exploration_rate']
        self.steps = checkpoint['steps']
        self.start_learning = checkpoint['start_learning']
        self.name = checkpoint['name']
        self.learn_per_n_steps = checkpoint['learn_per_n_steps']
        loss_fn_class = getattr(torch.nn, checkpoint['loss_fn'])
        self.loss_fn = loss_fn_class()
        self.lr = checkpoint['lr']
        self.replay_size = checkpoint['replay_size']
        self.memory = self.create_memory_buffer(replay_size=self.replay_size, device=self.device)


class PRDQN(DQN):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 1000,
                 batch_size = 64,
                 gamma = 0.95,
                 tau = 0.005,
                 lr = 1e-4,
                 start_learning=128,
                 replay_size=512,
                 loss_fn=nn.MSELoss,
                 name="PRDQN",
                 learn_per_n_steps=1,
                 buffer_device="cpu",
                 device="cpu"):
        super().__init__(input_shape,
                          output_dim,
                          eps_start = eps_start,
                          eps_end = eps_end,
                          eps_decay = eps_decay,
                          batch_size = batch_size,
                          gamma = gamma,
                          tau = tau,
                          lr = lr,
                          start_learning=start_learning,
                          replay_size=replay_size,
                          loss_fn=loss_fn,
                          name=name,
                          learn_per_n_steps=learn_per_n_steps,
                          buffer_device=buffer_device,
                          device=device)
        self.loss_fn = loss_fn(reduction="none")
    # Prioritized replay buffer
    def create_memory_buffer(self, replay_size, device):
        return TensorDictPrioritizedReplayBuffer(storage=LazyTensorStorage(replay_size, device=device), alpha=0.5, beta=0.4)
    
    def cache(self, obs, next_obs, action, action_num, reward, done):
        with torch.no_grad():
            td_estimate = self.online_net(obs).gather(1, action_num.unsqueeze(1))
            online_next_actions = torch.argmax(self.online_net(next_obs), axis=1)
            target_next_values = self.target_net(next_obs).gather(1, online_next_actions.unsqueeze(1))
            td_target = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * target_next_values
            td_error = self.loss_fn(td_estimate, td_target).item()
        if self.buffer_device != self.device:
            self.memory.add(TensorDict({
                "obs": obs.to(self.buffer_device), 
                "next_obs": next_obs.to(self.buffer_device), 
                "action": action.to(self.buffer_device), 
                "action_num": action_num.to(self.buffer_device), 
                "reward": reward.to(self.buffer_device), 
                "done": done.to(self.buffer_device),
                "td_error": td_error}, batch_size=[]))
        else:
            self.memory.add(TensorDict({
                "obs": obs, 
                "next_obs": next_obs, 
                "action": action, 
                "action_num": action_num, 
                "reward": reward, 
                "done": done,
                "td_error": td_error}, batch_size=[]))

    def recall(self, batch_size):
        batch = self.memory.sample(batch_size)
        obs, next_obs, action, action_num, reward, done, indices, weights = (batch.get(key) for key in ("obs", "next_obs", "action", "action_num", "reward", "done", "index", "_weight"))
        if self.buffer_device == self.device:
            return obs.squeeze(1), next_obs.squeeze(1), action, action_num, reward, done, indices, weights
        return obs.squeeze(1).to(self.device), next_obs.squeeze(1).to(self.device), action.to(self.device), action_num.to(self.device), reward.to(self.device), done.to(self.device), indices.to(self.device), weights.to(self.device)  
        
    def update_online(self):
        obs, next_obs, action, action_num, reward, done, indices, weights = self.recall(self.batch_size)
        td_estimate = self.online_net(obs).gather(1, action_num)
        with torch.no_grad():
            online_next_actions = torch.argmax(self.online_net(next_obs), axis=1)
            target_next_values = self.target_net(next_obs).gather(1, online_next_actions.unsqueeze(1))
        td_target = reward + (1 - done) * self.gamma * target_next_values
        td_error = self.loss_fn(td_estimate, td_target).flatten()
        loss = (weights * td_error).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priority(indices, td_error.detach())
        return loss.item()

class ValueAdvantageNet(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.conv_net, conv_out_size = self.create_conv_net(input_shape)
        self.value_net = self.create_value_net(conv_out_size)
        self.adv_net = self.create_adv_net(conv_out_size, output_dim)

    def create_conv_net(self, input_shape):
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
            nn.Flatten(), 32 * 11 * 11
        )

    def create_value_net(self, conv_out_size):
        return nn.Sequential(
            nn.Linear(conv_out_size, 1)
        )
    
    def create_adv_net(self, conv_out_size, output_dim):
        return nn.Sequential(
            nn.Linear(conv_out_size, output_dim)
        )
    
    def forward(self, data):
        conv = self.conv_net(data)
        value = self.value_net(conv)
        adv = self.adv_net(conv)
        return value + (adv - adv.mean(dim=1, keepdim=True))


class DuelingPRDQN(PRDQN):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 1000,
                 batch_size = 64,
                 gamma = 0.95,
                 tau = 0.005,
                 lr = 1e-4,
                 start_learning=128,
                 replay_size=512,
                 loss_fn=nn.MSELoss,
                 name="DuelingPRDQN",
                 learn_per_n_steps=1,
                 buffer_device="cpu",
                 device="cpu"):
        super().__init__(input_shape,
                          output_dim,
                          eps_start = eps_start,
                          eps_end = eps_end,
                          eps_decay = eps_decay,
                          batch_size = batch_size,
                          gamma = gamma,
                          tau = tau,
                          lr = lr,
                          start_learning=start_learning,
                          replay_size=replay_size,
                          loss_fn=loss_fn,
                          name=name,
                          learn_per_n_steps=learn_per_n_steps,
                          buffer_device=buffer_device,
                          device=device)
    def create_net(self, input_shape, output_dim):
        return ValueAdvantageNet(input_shape, output_dim)

class NstepDuelingPRDQN(DuelingPRDQN):
    def __init__(self,
                 input_shape,
                 output_dim=9,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 1000,
                 batch_size = 64,
                 gamma = 0.95,
                 tau = 0.005,
                 lr = 1e-4,
                 start_learning=128,
                 replay_size=512,
                 loss_fn=nn.MSELoss,
                 name="NstepDuelingPRDQN",
                 learn_per_n_steps=1,
                 n_steps=6,
                 buffer_device="cpu",
                 device="cpu"):
        super().__init__(input_shape,
                          output_dim,
                          eps_start = eps_start,
                          eps_end = eps_end,
                          eps_decay = eps_decay,
                          batch_size = batch_size,
                          gamma = gamma,
                          tau = tau,
                          lr = lr,
                          start_learning=start_learning,
                          replay_size=replay_size,
                          loss_fn=loss_fn,
                          name=name,
                          learn_per_n_steps=learn_per_n_steps,
                          buffer_device=buffer_device,
                          device=device)
        self.n_steps = n_steps
        self.n_step_buffer = deque()

    def cache(self, obs, next_obs, action, action_num, reward, done):
        # Accumulate n steps
        self.n_step_buffer.append((obs, next_obs, action, action_num, reward, done))
        if done.item() > 0 or len(self.n_step_buffer) >= self.n_steps:
            self.process_nstep_buffer()
            if done:
                self.n_step_buffer.clear()
            else:
                self.n_step_buffer.popleft()
    
    def process_nstep_buffer(self):        
        steps = min(self.n_steps, len(self.n_step_buffer))
        n_step_reward = sum(self.n_step_buffer[i][4] * (self.gamma ** i) for i in range(steps))
        n_step_done = self.n_step_buffer[-1][5] # if the traj ends with done

        init_step_obs = self.n_step_buffer[0][0]
        init_step_action = self.n_step_buffer[0][2]
        init_step_action_num = self.n_step_buffer[0][3]
        n_step_next_obs = self.n_step_buffer[-1][1]
        n_step_next_discount = self.gamma ** steps
        
        with torch.no_grad():
            # Estimate Q-value for first state action in traj
            td_estimate = self.online_net(init_step_obs).gather(1, init_step_action_num.unsqueeze(1)) 

            # Select next action after last state in traj
            online_next_actions = torch.argmax(self.online_net(n_step_next_obs), axis=1)  

            # Get the expected value for the next state from target network taking action provided by online net
            target_next_values = self.target_net(n_step_next_obs).gather(1, online_next_actions.unsqueeze(1))

            # Compute the n-step target using accumulated rewards  
            td_target = n_step_reward.unsqueeze(1) + (1 - n_step_done.unsqueeze(1)) * n_step_next_discount * target_next_values  

            td_error = self.loss_fn(td_estimate, td_target).item()  # Calculate the Temporal Difference (TD) error
  
        # Add the n-step transition to memory
        if self.device != self.buffer_device:
            self.memory.add(TensorDict({
                          "obs": init_step_obs.to(self.buffer_device),
                          "next_obs": n_step_next_obs.to(self.buffer_device),
                          "action": init_step_action.to(self.buffer_device),
                          "action_num": init_step_action_num.to(self.buffer_device),
                          "reward": n_step_reward.to(self.buffer_device),
                          "done": n_step_done.to(self.buffer_device),
                          "td_error": td_error.to(self.buffer_device), 
                          "step": torch.tensor([steps]).to(self.buffer_device)}, batch_size=[]))
        else:
            self.memory.add(TensorDict({
                "obs": init_step_obs,
                "next_obs": n_step_next_obs,
                "action": init_step_action,
                "action_num": init_step_action_num,
                "reward": n_step_reward,
                "done": n_step_done,
                "td_error": td_error, 
                "step": torch.tensor([steps])}, batch_size=[]))
        

    def recall(self, batch_size):
        batch = self.memory.sample(batch_size)
        obs, next_obs, action, action_num, reward, done, indices, weights, steps = (batch.get(key) for key in ("obs", "next_obs", "action", "action_num", "reward", "done", "index", "_weight", "step"))
        if self.device == self.buffer_device:
            return obs.squeeze(1), next_obs.squeeze(1), action, action_num, reward, done, indices, weights, steps
        return obs.squeeze(1).to(self.device), next_obs.squeeze(1).to(self.device), action.to(self.device), action_num.to(self.device), reward.to(self.device), done.to(self.device), indices.to(self.device), weights.to(self.device), steps.to(self.device)
    
        
    def update_online(self):
        obs, n_step_next_obs, action, action_num, reward, done, indices, weights, steps = self.recall(self.batch_size)
        td_estimate = self.online_net(obs).gather(1, action_num)
        with torch.no_grad():
            online_n_step_next_actions = torch.argmax(self.online_net(n_step_next_obs), axis=1)
            target_n_step_next_values = self.target_net(n_step_next_obs).gather(1, online_n_step_next_actions.unsqueeze(1))
        td_target = reward + (1 - done) * (self.gamma ** steps) * target_n_step_next_values
        td_error = self.loss_fn(td_estimate, td_target).flatten()
        loss = (weights * td_error).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priority(indices, td_error.detach())
        return loss.item()
    
    def make_save_obj(self, episode):
        obj = super().make_save_obj(episode)
        obj['n_steps'] = self.n_steps
        return obj

    def load_checkpoint(self, checkpoint):
        super().load_checkpoint(checkpoint)
        self.n_steps = checkpoint['n_steps']


class DQN_CARTPOLE(DQN):
    def __init__(self, input_shape, output_dim=2, eps_start=0.95, eps_end=0.05, eps_decay=1000, batch_size=64,
                  gamma=0.95, tau=0.005, lr=0.0001, start_learning=128, replay_size=512, loss_fn=nn.MSELoss,
                  name="DQN", learn_per_n_steps=1,
                  device="cuda", buffer_device="cuda"):
        super().__init__(input_shape, output_dim, eps_start, eps_end, eps_decay, batch_size, gamma, tau, lr, start_learning,
                          replay_size, loss_fn, name, learn_per_n_steps, device=device, buffer_device=buffer_device)

    def create_net(self, input_shape, output_dim):
        return nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def act(self, obs): 
        # select based on epsilon greedy policy (trade off between exploitation and exploration)
        if self.training and np.random.rand() < self.exploration_rate: # explore
            action_num = np.random.randint(self.output_dim)
            actions = [action_num]
        else: # exploit
            with torch.no_grad():
                obs = obs.to(device=self.device) # (1, C, H, W)
                action_values = self.online_net(obs)
                action_num = torch.argmax(action_values, axis=1).item()
                actions = [action_num]

        if self.training:
            self.update_exploration_rate()
            self.steps += 1
        return actions, action_num