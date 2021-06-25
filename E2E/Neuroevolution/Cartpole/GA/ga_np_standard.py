import numpy as np
import gym

import tensorflow as tf

from collections import deque
import copy
import time

from lr_schedular import CosineAnnealingSchedule
from model import PolicyModel

# STATICS

SAVE_PATH = './saved_elites'


# class to make the GA
def next_seed(current_seed, sampled_seed):
    return current_seed + sampled_seed


def save_elite(elite, generation_id):
    # call the model.save('PATH')
    elite['model'].save(SAVE_PATH, generation_id, elite['reward'])


class GeneticAlgorithm():
    def __init__(self, population_size, truncation_point, sigma, env,
                 num_generations, layer_hidden_shapes, elite_tournament_size) -> None:
        # Initialise a set of neural networks
        self.generation_id = 0
        self.elite_tournament_size = elite_tournament_size
        self.sigma = sigma  # mutation strength
        self.population_size = population_size

        # self.min_array = np.random.uniform(-0.00001, -0.01, population_size)
        # self.max_array = np.random.uniform(0.00001, 0.01, population_size)
        # self.seed = np.random.randint(0, 100, population_size-1)
        # make a seed table to store the seeds for each population in the generation
        seed_table = [
            sorted(
                np.random.randint(0, 2**16 - 1,
                                  size=num_generations, dtype=np.int), reverse=True) for _ in range(population_size - 1)
        ]
        self.seed_table = np.array(seed_table).T

        # make a table containing the truncation point indexes
        truncation_table = [[np.random.randint(0, truncation_point, dtype=np.int) for _ in range(population_size - 1)]
                            for _ in range(num_generations)]
        self.truncation_table = np.array(truncation_table)
        print(self.seed_table.shape)
        print(self.truncation_table.shape)
        # print(self.seed[0])
        self.model_population = deque(maxlen=population_size)
        self.elite_reward = 0
        # self.T = truncation_point
        self.num_steps = 1

        self.env = env
        # print(env.observation_space.shape)
        obs_shape = [env.observation_space.shape[0]]
        action_shape = [env.action_space.n]
        # print(obs_shape)
        self.layer_shapes = obs_shape + layer_hidden_shapes + action_shape
        # self.num_inputs =
        self.num_layers = len(self.layer_shapes) - 1
        self.scheduler = CosineAnnealingSchedule(0.0, sigma, num_generations)

    def evolve(self):
        self.sigma = self.scheduler(self.generation_id)
        self.generation_id += 1

        if self.generation_id == 1:
            for p in range(self.population_size - 1):
                # initialise the model weights
                policy_model = PolicyModel(self.layer_shapes)
                # _w, _b = policy_model.get_weights_biases()
                # print(_w.shape)
                # print(_b.shape)
                policy_model.init_weights('standard', self.seed_table[0][p])
                self.model_population.append({
                    'model': policy_model,
                    'seed': self.seed_table[0][p],
                    'reward': -21,
                })

        else:
            # get a new mutated population
            # before mutating make a copy of the deque as a list
            model_population_list = copy.deepcopy(self.model_population)
            # clear the deque for new population
            self.model_population.clear()
            # put back the elite
            self.model_population.insert(0, model_population_list[0])
            # set the deque with the new population
            for i in range(0, self.population_size - 1):
                k = self.truncation_table[self.generation_id - 1][i]
                # print(k)
                selected_individual = model_population_list[k]
                mutated_individual = self.mutate_individual(selected_individual, k)
                # clear the old population and make space for the new population
                self.model_population.append(mutated_individual)

        # print(self.model_population)
        print("Population size", len(self.model_population))
        population_mean_fitness = self.compute_fitness()

        return population_mean_fitness, self.sigma

    def mutate_individual(self, selected_individual, individual_index):
        # get the model weights and biases
        selected_model = selected_individual['model']
        current_seed = selected_individual['seed']
        current_reward = selected_individual['reward']

        # param_mean = [tf.reduce_mean(p).numpy() for p in selected_model.weights]

        new_seed = next_seed(current_seed, self.seed_table[self.generation_id - 1][individual_index])
        # print()
        # print("Seed", new_seed)
        # print()
        np.random.seed(new_seed)
        # get the current params
        w, b = selected_model.get_weights_biases()

        # make a new standar normal vector that of the shape of weight and biases
        normal_w = np.random.normal(size=len(w))
        normal_b = np.random.normal(size=len(b))
        # mutate the individual
        w = w + self.sigma * normal_w
        b = b + self.sigma * normal_b
        # assign the new weights to the selected model
        selected_model.set_weights_biases(w, b)
        '''
        for i in range(3, len(selected_model.weights)):
            params = selected_model.weights[i]
            ## mutate the individual
            #print(build_model(new_seed).weights[i])
            #print(params)
            new_params = params + self.sigma * tf.random.normal((params.shape),
                                                        mean=0.0,
                                                        stddev=1.0,
                                                        seed=new_seed) 
            selected_model.weights[i] = new_params
        '''
        # make a dict for storage
        mutated_individual = {
            'model': selected_model,
            'seed': new_seed,
            'reward': current_reward
        }

        return mutated_individual

    def play_episode(self, model):
        # episode loop
        current_obs = self.env.reset()
        # print(np.expand_dims(self.current_obs, 0).shape)
        current_obs = np.expand_dims(current_obs, 0)
        # print(tf.reduce_max(self.current_obs))
        # print(tf.reduce_min(self.current_obs))

        reward_list = list()

        while True:
            # take an action
            action = np.argmax(model(current_obs), -1)
            self.env.render()
            # print(action)
            # action = tf.argmax(action_probs, 1)[0].numpy()
            # print(action)
            # action = ACTION_MAP[action]

            next_obs, reward, done, _ = self.env.step(action[0])
            next_obs = next_obs
            current_obs = np.expand_dims(next_obs, 0)

            reward_list.append(reward)

            if done:
                break
        # print(sum(reward_list))
        return np.sum(reward_list), len(reward_list)  # np.random.uniform(21, -21, 1)[0]

    def compute_fitness(self):

        population_fitness = list()
        start = 0 if self.generation_id == 1 else 1
        print(start)
        for i in range(start, len(self.model_population)):
            # evaluate the model
            print("Gen, ", self.generation_id)
            print("Indi", i)
            print("Current fitness: ", self.model_population[i]['reward'])
            total_reward, self.num_steps = self.play_episode(
                self.model_population[i]['model'])
            self.model_population[i]['reward'] = total_reward
            # print("New mean fitness: ", self.model_population[i]['reward'])
            print("New fitness: ", self.model_population[i]['reward'])
            print("Num steps: ", self.num_steps)
            print()
            # print(total_reward)
            population_fitness.append(total_reward)

        mean_population_reward = np.mean(population_fitness)
        # elite_reward = self.model_population[0]['reward']
        print(mean_population_reward)
        return mean_population_reward

    def elite_tournament(self, best_n):
        for i in range(self.elite_tournament_size):
            total_reward = 0
            for j in range(30):
                episode_reward, _ = self.play_episode(best_n[i]['model'])
                total_reward += episode_reward
            mean_tournament_reward = total_reward / 30.0
            # reconstruct model dict
            best_n[i]['reward'] = mean_tournament_reward

            return best_n

    def select_elite(self, best_n):
        best_n_rewards = [m['reward'] for m in best_n]
        sorted_index_before = np.argsort(best_n_rewards)[::-1]
        print("Reward before tournament: ", best_n_rewards)
        print("Indices before tournament: ", sorted_index_before)
        best_n = self.elite_tournament(best_n)
        # sort the best 10 according to the new performances
        rewards = [m['reward'] for m in best_n]
        sorted_index = np.argsort(rewards)[::-1]
        # sorted_rewards = sorted(rewards, reverse=True)
        print("Reward after tournament: ", rewards)
        print("Indices after tournament: ", sorted_index)
        elite_index = sorted_index[0]
        # get the best model
        elite = best_n[elite_index]

        return elite, elite_index

    def sort_population(self):
        start = 0 if self.generation_id == 1 else 1
        print(start)
        if start == 0:
            # set the current elite
            elite_original = None
            self.model_population = deque(sorted(
                self.model_population, key=lambda x: x['reward'], reverse=True),
                maxlen=self.population_size)
        else:
            # make a copy of the elite
            elite_original = self.model_population[0]
            elite_original_reward = elite_original['reward']
            population_only = [self.model_population[i] for i in range(1, self.population_size)]
            self.model_population = deque(sorted(
                population_only, key=lambda x: x['reward'], reverse=True),
                maxlen=self.population_size)

        # print(elite)
        sorted_rewards = [m['reward'] for m in self.model_population]
        print(sorted_rewards)
        # If its the 1st generation select 10 individuals and play 30 games to
        # determine the correct elite
        if self.generation_id == 1:
            print()
            print("Evaluating Elites at gen 1")
            print()
            # get the best 10 if its the initial generation
            best_n = [copy.deepcopy(self.model_population[i]) for i in range(self.elite_tournament_size)]
            # select the elite
            elite, elite_index = self.select_elite(best_n)
            # remove the elite from the sorted list of model population
            # del self.model_population[elite_index]
            self.model_population.insert(0, elite)

            sorted_rewards = [m['reward'] for m in self.model_population]
            print("Final population reward: ", sorted_rewards)
            print(len(self.model_population))

        else:
            print()
            print("Evaluating Elites at gen {}".format(self.generation_id))
            best_n = [copy.deepcopy(self.model_population[i]) for i in range(self.elite_tournament_size - 1)]
            best_n.insert(0, elite_original)
            # select the elite
            elite, elite_index = self.select_elite(best_n)
            if elite_original['reward'] > elite['reward']:
                self.model_population.insert(0, elite_original)
            else:
                self.model_population.insert(0, elite)
            sorted_rewards = [m['reward'] for m in self.model_population]
            print("Final population reward: ", sorted_rewards)
            print(len(self.model_population))

        # save the elites who satisfies a threshold
        elites_to_save = [self.model_population[i] for i in range(5)]
        for elite in elites_to_save:
            if elite['reward'] > 100:
                print("Saving Elite")
                save_elite(elite, self.generation_id)
        # return the elite reward for plotting
        return self.model_population[0]['reward']


def main(population_size, truncation_point, sigma):
    # env = gym.make('CartPole-v0')
    gym.envs.register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=1000,
        reward_threshold=950.0,
    )
    env = gym.make('CartPole-v2')
    num_generations = 40
    layer_shapes = [32, 32]
    ga_optimiser = GeneticAlgorithm(
        population_size,
        truncation_point,
        sigma,
        env, num_generations, layer_shapes, 10)

    writer = tf.summary.create_file_writer('./np_logs/cartpole_elite_saves')
    with writer.as_default():
        for ep in range(num_generations):
            print()
            print("##### NEW POPULATION #####")
            print()
            population_reward, sigma_step = ga_optimiser.evolve()
            elite_reward = ga_optimiser.sort_population()
            tf.summary.scalar('population mean reward', population_reward, ep + 1)
            tf.summary.scalar('Elite reward', elite_reward, ep + 1)
            tf.summary.scalar('sigma decay', sigma_step, ep + 1)
            print()
            print("Population Mean: ", population_reward)
            print()


if __name__ == "__main__":
    main(500, 30, 0.0002)
'''
#rewards = np.random.randint(0.0, 10.0, 10)
ga_optimiser.evolve()
ga_optimiser.sort_population()
print()
print("Gen 2")
print()
#rewards = np.random.randint(0.0, 10.0, 10)
ga_optimiser.evolve()
ga_optimiser.sort_population()
print()
print("Gen 3")
print()
ga_optimiser.evolve()
ga_optimiser.sort_population()
'''

'''
ga_optimiser.sort_population(rewards)

'''
