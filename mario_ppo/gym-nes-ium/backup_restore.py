from tqdm import tqdm

from nes_py.NESEnv import MockNESEnv

if __name__ == "__main__":
    env = MockNESEnv('./nes_py/tests/games/super-mario-bros-1.nes')
    env.reset()

    for i in tqdm(range(5_000)):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        if (i + 1) % 12:
            env._backup()
        if (i + 1) % 27:
            env._restore()

        if done:
            env.reset()
