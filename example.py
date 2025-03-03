import retro

def main():
    env = retro.make(game='IceClimber-Nes')
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info  = env.step(action)
        env.render()
        if terminated:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
