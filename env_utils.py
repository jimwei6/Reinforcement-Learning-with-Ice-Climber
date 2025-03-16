# Code reference from pytorch
import gymnasium as gym
import torchvision.transforms as T
import random

# Basic frame skip - apply same action per n amount of frames
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip_frames=1):
        super().__init__(env)
        self.skip_frames = skip_frames

    def step(self, action):
        terminated = False
        truncated = False
        total_rewards = 0
        for i in range(self.skip_frames):
            ob, reward, terminated, truncated, info = self.env.step(action)

            total_rewards += reward
            if terminated or truncated:
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
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = random.randint(0, self.noop_max)

        for i in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.env.action_space.sample())
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info