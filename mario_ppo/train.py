import gymnasium
# from gymnasium.
import stable_baselines3
from stable_baselines3 import DQN

env = gymnasium.make("ALE/MarioBros-ram-v5", render_mode='human')
# env = gymnasium.wrappers.

model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)
model.save("dqn_mario")

# del model

model = DQN.load('dqn_mario')

obs, info = env.reset()

done = True
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()