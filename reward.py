import numpy as np

# Sparse - only contain rewards related to dying, height change, and hammer hits
class SparseRewardTracker():
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

    def update(self, info, next_info, truncated, terminated, action, delta):
        new_delta = delta.copy()
        self.curr_ep_length += 1
        self.curr_ep_max_height = max(self.curr_ep_max_height, next_info['height'])
        new_delta['hammer_hit'] = False if action[0] == 0 else True
        new_delta['jump'] = False if action[-1] == 0 else True
        new_delta['right'] = False if action[-2] == 0 else True
        new_delta['left'] = False if action[-3] == 0 else True
        new_delta['still'] = not new_delta['hammer_hit'] and not new_delta['jump'] and not new_delta['right'] and not new_delta['left']
        new_delta['landed'] = False
        new_delta['jump_net_height'] = 0
    
        if len(info) > 0:
            # Track effective landings
            if not self.jump and action[-1] == 1 and next_info['in_air'] == 1 and info['height'] > 0: 
                self.jump = True
                self.jump_start_height = info['height']
            elif self.jump and next_info['in_air'] == 0: # landed
                self.jump = False
                new_delta['landed'] = True
                new_delta['jump_net_height'] = next_info['height'] - self.jump_start_height
                self.jump_start_height = 0

        return new_delta

    def calculate_reward(self, info, next_info, truncated, terminated, action):      
        delta = self.calculate_info_delta(info, next_info)
        delta = self.update(info, next_info, truncated, terminated, action, delta)

        landing_reward = 0 # signal for landing on jumps
        alive_reward = 0 # signal for living / dying
        height_reward = 0 # signal for changing height
        hammer_penalty = -0.01 if delta['hammer_hit'] else 0
        
        if terminated:
            alive_reward = -1
        else:
            alive_reward = (next_info['height'] )**2 / 100 if not delta['still'] else 0 # parabolic increase from [0.01 (height 1) - 1 (height 10)]
            if delta['landed']: 
                if delta['jump_net_height'] > 0: # effective jumps
                    landing_reward = 1
                elif delta['jump_net_height'] < 0: # jumping downwards
                    landing_reward = -0.01
                        
        rew = landing_reward + height_reward + alive_reward + hammer_penalty
        return max(min(rew, 1), -1)
    
class IntermediateRewardTracker(SparseRewardTracker):
    def __init__(self):
        super().__init__()

    def calculate_reward(self, info, next_info, truncated, terminated, action):
        delta = self.calculate_info_delta(info, next_info)
        delta = self.update(info, next_info, truncated, terminated, action, delta)

        landing_reward = 0 # signal for landing on jumps
        alive_reward = 0 # signal for living / dying
        height_reward = 0 # signal for successfully changing height
        hammer_penalty = -0.01 if delta['hammer_hit'] else 0
        brick_reward = 0.05 if delta['bricks_hit'] > 0 else 0
        change_height_reward = 0 # intermediate height change signal

        if terminated:
            alive_reward = -1
        else:
            alive_reward = (next_info['height'] )**2 / 100 if not delta['still'] else 0 
            if delta['landed']: 
                if delta['jump_net_height'] > 0: # effective jumps
                    landing_reward = 1
            
            if delta['height'] > 0: # intermediate height change reward
                change_height_reward = 1
            elif delta['height'] < 0: # intermediate fall change reward
                change_height_reward = -0.5
                 
        rew = landing_reward + height_reward + alive_reward + hammer_penalty + brick_reward + change_height_reward
        return max(min(rew, 1), -1)