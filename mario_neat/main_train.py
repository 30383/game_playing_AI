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

class Train:
    def __init__(self, generations, parallel=16, level='1-1'):
        self.generations = generations
        self.lock = mp.Lock()
        self.par = parallel
        self.level = level
    
    def _eval_genomes_no_paralle(self, genomes, config):
        ENV_NAME = 'SuperMarioBros-1-1-v3'
        env = gym_super_mario_bros.make(ENV_NAME, render_mode = 'ansi', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = apply_wrappers(env)
        for genome in genomes:
            try:
                state, _ = env.reset()
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                done = False
                i = 0
                old_x_pos = 40
                fitness = 0
                while not done:
                    state = state.__array__()
                    state = state.flatten()
                    output = net.activate(state)
                    s, reward, done, trun, info = env.step(output.index(max(output)))
                    fitness += reward
                    state = s
                    i += 1
                    if i%50 == 0:
                        if old_x_pos == info['x_pos']:
                            fitness -= 50
                            break
                        else:
                            old_x_pos = info['x_pos']
                            
                    genome.fitness += fitness
                    env.close()
            except KeyboardInterrupt:
                env.close()
                exit()
    
    def _eval_genome(self, genome, config, o):
        ENV_NAME = 'SuperMarioBros-1-1-v3'
        env = gym_super_mario_bros.make(ENV_NAME, render_mode = 'ansi', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = apply_wrappers(env)
        try:
            state, _ = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            i = 0
            fitness = 0
            old_x_pos = 40
            while not done:
                state = state.__array__()
                state = state.flatten()
                output = net.activate(state)
                s, reward, done, trun, info = env.step(output.index(max(output)))
                fitness += reward
                state = s
                i += 1
                if i%50 == 0:
                    if old_x_pos == info['x_pos']:
                        fitness -= 50
                        break
                    else:
                        old_x_pos = info['x_pos']

            if fitness >= 3252:
                pickle.dump(genome, open("finisher.pkl", "wb"))
                env.close()
                print("Done")
                exit()
            o.put(fitness)
            env.close() 
        except KeyboardInterrupt:
                env.close()
                exit()       
                
                
    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)
        
        for i in range(0, len(genomes), self.par):
            output = mp.Queue()
            
            processes = [mp.Process(target=self._eval_genome, args=(genome, config, output)) for genome in genomes[i:i + self.par]]
            
            [p.start() for p in processes]
            [p.join() for p in processes]
            
            results = [output.get() for p in processes]
            
            for n, r in enumerate(results):
                genomes[i+n].fitness = r

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        winner = p.run(self._eval_genomes, n)
        win = p.best_genome
        pickle.dump(winner, open('winner.pkl', 'wb'))
        pickle.dump(win, open('real_winner.pkl', 'wb'))
        
        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        
    def main(self, config_file='config.txt'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)
        
if __name__ == "__main__":
    t = Train(500)
    t.main()      
            
            