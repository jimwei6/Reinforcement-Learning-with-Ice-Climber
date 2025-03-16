# Referenced implementations from stable-retro, pytorch mario tutorial
import numpy as np
import json
import os

class Logger():
  def __init__(self):
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
              json.dump({"ep_rewards": [], "mean_ep_loss": [],
                          "ep_length": [], "ep_max_level": [],
                          "ep_ending": [], "ep_final_info": [],
                          "ep_action_summary": []}, json_file, indent=4)

  def save(self):
      log_data = {
          "ep_rewards": self.ep_rewards,
          "mean_ep_loss": self.mean_ep_loss,
          "ep_length": self.ep_length,
          "ep_max_level": self.ep_max_height,
          "ep_ending": self.ep_ending,
          "ep_final_info": self.ep_final_info,
          "ep_action_summary": self.ep_action_summary
      }
        
      with open(self.log_path, "w") as json_file:
          json.dump(log_data, json_file, indent=4)
      
  def log_episode(self, final_info, ending="truncated"):
      self.ep_rewards.append(self.curr_ep_reward)
      self.ep_length.append(self.curr_ep_length)
      self.mean_ep_loss.append(np.mean(self.curr_ep_loss))
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

  def reset_episode(self):
      self.curr_ep_reward = 0
      self.curr_ep_length = 0
      self.curr_ep_loss = []
      self.curr_ep_max_height = 1
      self.curr_action_summary = np.array([0] * 4)
      