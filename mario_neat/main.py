import os
import gym_super_mario_bros
import neat
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import numpy as np
import pickle

CONFIG = 'config.txt'

def main(config_file, file, level = '1-1'):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    ENV_NAME = 'SuperMarioBros-1-1-v0'
    env = gym_super_mario_bros.make(ENV_NAME, render_mode = 'human', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = {'x_pos': 0}
    state, _ = env.reset()
    while info['x_pos'] != 3253:
        
        done = False
        i = 0
        old = 40
        while not done:
            state = state.__array__()
            state = state.flatten
            action = net.activate(state)
            s, reward, done, _, info = env.step(action.index(max(action)))
            state = s
            i += 1
            if i % 100 == 0:
                if old == info['x_pos']:
                    break
                else:
                    old = info['x_pos']
        print("Distance: {}".format(info['x_pos']))
    env.close()
    
if __name__ == "__main__":
    main(CONFIG, "real_winner.pkl")