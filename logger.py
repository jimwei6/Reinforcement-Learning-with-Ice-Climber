# Referenced implementations from stable-retro, pytorch mario tutorial
import numpy as np
import json
import os

class Logger():
  def __init__(self):
      pass

  def create_save_data(self):
      pass

  def save(self):
      pass
  
  def log_episode(self):
      pass
     
  def log_step(self):
      pass

  def reset_episode(self):
      pass

class DQNLogger(Logger):
  def __init__(self, log_path):
      self.ep_rewards = []
      self.mean_ep_loss = []
      self.ep_length = []
      self.ep_max_height = []
      self.ep_ending = []
      self.ep_final_info = []
      self.ep_action_summary = []
      self.reset_episode()
      self.log_path = log_path
      
      log_dir = os.path.dirname(self.log_path)
      if log_dir:
          os.makedirs(log_dir, exist_ok=True)

      if not os.path.exists(self.log_path):
          with open(self.log_path, "w") as json_file:
              json.dump(self.create_save_data(), json_file, indent=4)

  def create_save_data(self):
      return {
          "ep_rewards": self.ep_rewards,
          "mean_ep_loss": self.mean_ep_loss,
          "ep_length": self.ep_length,
          "ep_max_level": self.ep_max_height,
          "ep_ending": self.ep_ending,
          "ep_final_info": self.ep_final_info,
          "ep_action_summary": self.ep_action_summary
      }

  def save(self):
      log_data = self.create_save_data()
      with open(self.log_path, "w") as json_file:
          json.dump(log_data, json_file, indent=4)
      
  def log_episode(self, final_info, ending="truncated"):
      self.ep_rewards.append(self.curr_ep_reward)
      self.ep_length.append(self.curr_ep_length)
      self.mean_ep_loss.append(np.mean(self.curr_ep_loss) if len(self.curr_ep_loss) else None)
      self.ep_max_height.append(self.curr_ep_max_height)
      self.ep_ending.append(ending)
      self.ep_final_info.append(final_info)
      self.ep_action_summary.append(self.curr_action_summary.tolist())
      self.reset_episode()
      self.save()
     
  def log_step(self, loss, reward, height, action):
      if loss is not None:
          self.curr_ep_loss.append(loss)
      self.curr_ep_length += 1
      self.curr_ep_reward += reward
      self.curr_ep_max_height = max(self.curr_ep_max_height, height)
      self.curr_action_summary += np.array(action[-4:])
      self.curr_action_summary[0] += action[0]

  def reset_episode(self):
      self.curr_ep_reward = 0
      self.curr_ep_length = 0
      self.curr_ep_loss = []
      self.curr_ep_max_height = 1
      self.curr_action_summary = np.array([0] * 4)

# Epoch based compared to DQN (episode based)
class PGLogger(DQNLogger):
    def __init__(self, log_path):
        self.ep_avg_probs = []
        self.batch_avg_probs = []
        self.batch_rewards = []
        self.mean_batch_loss = []
        self.batch_length = []
        self.batch_max_height = []
        self.batch_min_height = []
        self.batch_ending = []
        self.batch_action_summary = []
        super().__init__(log_path)


    def log_batch(self, loss):
        self.batch_rewards.append(np.mean(self.ep_rewards).item())
        self.batch_length.append(np.mean(self.ep_length).item())
        self.mean_batch_loss.append(loss)
        self.batch_max_height.append(np.max(self.ep_max_height).item())
        self.batch_min_height.append(np.min(self.ep_max_height).item())
        self.batch_ending.append(self.ep_ending.copy())
        self.batch_action_summary.append(np.mean(self.ep_action_summary, axis=0).tolist())
        self.batch_avg_probs.append(np.mean(self.ep_avg_probs, axis=0).tolist())
        self.save()
        self.reset_batch()

    def reset_batch(self):
        self.ep_rewards = []
        self.ep_length = []
        self.ep_max_height = []
        self.ep_ending = []
        self.ep_action_summary = []
        self.ep_avg_probs = []

    def log_episode(self, ending="truncated"):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_length.append(self.curr_ep_length)
        self.ep_max_height.append(self.curr_ep_max_height)
        self.ep_ending.append(ending)
        self.ep_action_summary.append(self.curr_action_summary.tolist())
        self.ep_avg_probs.append((self.ep_probs / self.curr_ep_length).tolist())
        self.reset_episode()

    def reset_episode(self):
        super().reset_episode()
        self.ep_probs = np.zeros(9)

    def log_step(self, loss, reward, height, action, probs):
        super().log_step(loss, reward, height, action)
        self.ep_probs += probs

    def create_save_data(self):
        data = {
            "batch_mean_rewards": self.batch_rewards,
            "batch_mean_loss": self.mean_batch_loss,
            "batch_mean_length": self.batch_length,
            "batch_max_level": self.batch_max_height,
            "batch_min_max_level": self.batch_min_height,
            "batch_ending": self.batch_ending,
            "batch_mean_action_summary": self.batch_action_summary,
            "batch_mean_action_probs": self.batch_avg_probs
        }
        return data

        
      