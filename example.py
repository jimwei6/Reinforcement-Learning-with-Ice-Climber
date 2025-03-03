import retro

def main():
    env = retro.make(game='IceClimber-Nes')
    obs = env.reset()
    while True:
        obs, rew, terminated, truncated, info  = env.step(env.action_space.sample())
        env.render()
        if terminated:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
