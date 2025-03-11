# Code reference from pytorch
import gymnasium as gym
import torchvision.transforms as T

# Basic frame skip - apply same action per n amount of frames
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip_frames=1):
        gym.Wrapper.__init__(self, env)
        self.skip_frames = skip_frames

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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

        