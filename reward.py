import numpy as np

class RewardTracker():
  def __init__(self):
      self.reset()
      
  def calculate_info_delta(self, info, next_info):
      if info is None or len(info) == 0: # initial step 
          next_info['height'] -= 1
          return next_info
      delta = {}
      for k, v in info.items():
          delta[k] = next_info[k] - info[k]
      return delta

  def reset(self):
      self.curr_ep_length = 0
      self.curr_ep_max_height = 1
      self.jump = False
      self.total_jumps = 0
      self.total_hits = 0
      self.height_bricks_hit = {}
      self.height_jumps = {}
      self.height_hammer_hits = {}
      for i in range(20):
          self.height_bricks_hit[i] = 0
          self.height_jumps[i] = 0
          self.height_hammer_hits[i] = 0

  def update(self, info, next_info, truncated, terminated, action, delta):
      new_delta = delta.copy()
      self.curr_ep_length += 1
      self.curr_ep_max_height = max(self.curr_ep_max_height, next_info['height'])
      self.in_air = next_info['in_air']

      new_delta['hammer_hit'] = False
      # track number of hits regardless if it was viable or not. Holding spams so keep increasing hits
      if action[0] == 1:
          self.total_hits += 1
          new_delta['hammer_hit'] = True

      new_delta['landed'] = False
      new_delta['jump_net_height'] = 0
      if len(info) > 0:
          # track number of jumps
          if not self.jump and action[-3] == 1 and info['in_air'] == 0: # if jump is started on the ground
              self.total_jumps += 1
              self.jump = True
              self.jump_start_height = info['height']
              self.height_jumps[info['height']] += 1

          if self.jump and next_info['in_air'] == 1: # landed
              self.jump = False
              new_delta['landed'] = True
              new_delta['jump_net_height'] = next_info['height'] - self.jump_start_height
              self.jump_start_height = 0
      return new_delta

  def calculate_reward(self, info, next_info, truncated, terminated, action):      
      delta = self.calculate_info_delta(info, next_info)
      curr_ep_length = self.curr_ep_length
      curr_ep_max_height = self.curr_ep_max_height

      # update
      delta = self.update(info, next_info, truncated, terminated, action, delta)  
      if terminated or delta['lives'] < 0: 
          if next_info['lives'] < 0 or delta['lives'] < 0:
              # penalize death more if it is at lower height
              return -10 - max(10 - next_info['height'], 0)
          elif next_info['height'] >= 10:
              # penalize excess jumps (average jumps per level - 10)
              penalize_jumps = min((self.total_jumps / 10) - 10, 5)
              # penalize excess hits 
              penalize_hits = min((self.total_hits / 10) - 10, 5)
              return max(10, 20 - penalize_jumps - penalize_hits) 
      else:
          rew = 0
          height = info['height'] if len(info) > 1 else 1

          # hitting rewards
          if delta['bricks_hit'] > 0: # a brick was hit
              self.height_bricks_hit[height] += 1
              if self.height_bricks_hit[height] <= 10: # decaying rewards after hitting max amount of bricks per level
                  rew += 2
              else:
                  rew += (2) * np.exp(- (self.height_bricks_hit[height] - 10) / 5)

          # These rewards are quite sparse so no limits applied
          if delta['birds_hit'] > 0: # a bird was hit
              rew += 5
          if delta['ice_hit'] > 0: # ice was hit
              rew += 5
          # landed after jumping
          if delta['landed']: 
              if delta['jump_net_height'] > 0: # effective jump
                  rew += 8
              elif delta['jump_net_height'] < 0: # negative jump
                  rew += -5
              else:
                  rew += -1

          # Punish for using the hammer
          if delta['hammer_hit']: 
              rew += -5 + 2 * np.exp(-self.height_hammer_hits[height]/50)

          if delta['height'] > 0: # jumped or moved up (in the air)
              if next_info['height'] <= curr_ep_max_height: # decay based on number of jumps in the level to prevent repeat jumping
                  rew += -1 + 3 * np.exp(-self.height_jumps[height]/5)  
              else:
                  return 5 # going to new height is highly encouraged
          elif delta['height'] < 0: # fell down or jump was ineffective
              rew += -5
          else: # no change in height
              if next_info['height'] == curr_ep_max_height: # encourages change in height but not too much to allow horizontal movements
                  rew += -5
              else: # character is below max height encourage getting back
                  rew += -5
          return max(-10, min(10, rew)) / 10 # keep rewards between [-1, 1]