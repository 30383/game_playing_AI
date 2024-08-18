import os
import visualize
import gym_super_mario_bros
import neat
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import numpy as np
import multiprocessing as mp
import pickle
import warnings 
warnings.filterwarnings("ignore")

# fitarray_global = []
# fitarray_local = []
parallel = 8

def eval_genomes_no_parallel(genomes, config):
    
    ENV_NAME = 'SuperMarioBros-1-1-v3'
    env = gym_super_mario_bros.make(ENV_NAME, render_mode = 'human', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env)
    env.reset()
    nets = []
    
    
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        state, _ = env.reset()
    
        #print(np.shape(state))
        
        done = False
        
        i = 0
        old = 40
        fitness = 0.0
        
        while not done:
            
            state = state.__array__()
            
            state = state.flatten()
            
            action = net.activate(state)
            
            state, reward, done, _,info = env.step(action.index(max(action)))
            
            fitness += reward
            
            i += 1
            if i%50 == 0:
                if old == info['x_pos']:
                    fitness -= 50
                    break
                else:
                    old = info['x_pos']
            # if gym_super_mario_bros.smb_env.SuperMarioBrosEnv._is_dead:
            #     fitness -= 25

            # Update the genome's fitness based on the reward or game-specific criteria
        
        genome.fitness += fitness
        print(fitness)
        
def eval_genome_parallel(genome, config, o):
    
    ENV_NAME = 'SuperMarioBros-1-1-v3'
    env = gym_super_mario_bros.make(ENV_NAME, render_mode = 'human', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env)
    env.reset()
    
    genome.fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state, _ = env.reset()

    #print(np.shape(state))
    
    done = False
    
    i = 0
    old = 40
    fitness = 0.0
    
    while not done:
        
        state = state.__array__()
        
        state = state.flatten()
        
        action = net.activate(state)
        
        state, reward, done, _,info = env.step(action.index(max(action)))
        
        fitness += reward
        
        i += 1
        if i%50 == 0:
            if old == info['x_pos']:
                fitness -= 50
                break
            else:
                old = info['x_pos']
        # if gym_super_mario_bros.smb_env.SuperMarioBrosEnv._is_dead:
        #     fitness -= 25

        # Update the genome's fitness based on the reward or game-specific criteria
    genome.fitness += fitness
    o.put(fitness)
    print(fitness)
    
def eval_genomes(genomes, config):
    idx, genomes = zip(*genomes)
    
    for i in range(0, len(genomes), parallel):
        output = mp.Queue()
        
        processes = [mp.Process(target=eval_genome_parallel, args=(genome, config, output)) for genome in genomes[i:i + parallel]]
        
        [p.start() for p in processes]
        [p.join() for p in processes]
        
        results = [output.get() for p in processes]
        
        for n, r in enumerate(results):
            genomes[i + n].fitness = r

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(eval_genomes, 5)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    pickle.dump(winner, open('winner.pkl', 'wb'))
    pickle.dump(winner_net, open('real_winner.pkl', 'wb'))
    
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    
def main(config_file='git_config.txt'):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    run(config_path)   

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    main()
