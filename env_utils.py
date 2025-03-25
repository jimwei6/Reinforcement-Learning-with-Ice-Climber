# Code reference from pytorch
import gymnasium as gym
import torchvision.transforms as T
import random
from gymnasium.wrappers.frame_stack import FrameStack
import torch

class FrameStackMod(FrameStack):
    def __init__(self, env, num_stack, lz4_compress = False):
        super().__init__(env, num_stack, lz4_compress)

    def observation(self, observation):
        lazyframes = super().observation(observation)
        return torch.tensor(lazyframes.__array__())

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_tracker):
        super().__init__(env)
        self.reward_tracker = reward_tracker

    def step(self, action, info):
        ob, _, terminated, truncated, next_info = self.env.step(action)
        reward = self.reward_tracker.calculate_reward(info, next_info, truncated, terminated, action)
        return ob, reward, terminated, truncated, next_info

# Basic frame skip - apply same action per n amount of frames
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip_frames=1):
        super().__init__(env)
        self.skip_frames = skip_frames

    def step(self, action, info):
        total_rewards = 0
        for i in range(self.skip_frames):
            ob, reward, terminated, truncated, next_info = self.env.step(action, info)
            total_rewards += reward
            info = next_info
            if terminated or truncated or next_info['lives'] < 3:
                break
        return ob, total_rewards, terminated, truncated, info
  
# convert rgb to gray scale
class GrayEnvironment(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        grayscale = T.Grayscale()
        observation = grayscale(observation)
        return observation[0]
    
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=None):
        super().__init__(env)
        self.shape = shape

    def observation(self, observation):
        compose = [T.ToTensor()]
        if self.shape:
            compose.append(T.Resize(self.shape))
        transforms = T.Compose(compose)
        observation = transforms(observation)
        return observation

# run random actions when an environment is reset (helps with random initialization) references pytorch & stablebaselines
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=50):
        super().__init__(env)
        self.noop_max = noop_max

    def convert_nums_to_actions(self, num):
        if num == 8:     # 8 means hit
            return [1, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            return [0, 0, 0, 0, 0, 0] + [int(x) for x in format(num, '03b')]
    

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = random.randint(0, self.noop_max)

        for i in range(noops):
            action_num = random.randint(0, 8)
            obs, _, terminated, truncated, info = self.env.step(self.convert_nums_to_actions(action_num))
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info