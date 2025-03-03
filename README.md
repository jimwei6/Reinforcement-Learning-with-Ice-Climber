# Reinforcement Learning for Ice Climber Retro Game

## Setup

Clone this repository and then follow stable-retro (https://github.com/Farama-Foundation/stable-retro) for set up. Read the installation and example sections. Then, copy files in IceClimber-Custom into "stable-retro/reto/data/contrib/IceClimber-Nes" directory to import the modified variables and environment files.

Run the following to register the environment in the module
  ```
  pip -m retro.import ./stable-retro/reto/data/contrib/IceClimber-Nes
  ``` 

Test the environment using example.py. This will load a window of Ice Climber where the agent performs random actions.
```
python ./example.py
```

## References

- Retro (https://github.com/openai/retro)
- Stable-Retro (https://github.com/Farama-Foundation/stable-retro)
- CleanRL (https://docs.cleanrl.dev/#citing-cleanrl)