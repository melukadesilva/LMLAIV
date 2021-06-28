########################## THIS CODE IS BASED ON ################################################
# https://github.com/alirezamika/evostra.git
# https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
# https://openai.com/blog/evolution-strategies/
 
import numpy as np
import multiprocessing as mp
from model import PolicyModel


def worker_process(worker_args):
    func, params = worker_args
    return func(params)


class NaturalES:
    def __init__(self, sigma, model, population_size, save_path, reward_function=None,
                 num_iterations=1000, alpha=0.01, num_threads=1, print_step=10,
                 save_point=1000):
        self.SIGMA = sigma
        self.save_point = save_point
        self.save_path = save_path
        self.model = model
        self.reward_function = reward_function
        self.parameters = list(self.model.get_weights_biases())
        self.population_size = population_size
        self.alpha = alpha
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.num_iterations = num_iterations
        self.print_step = print_step

    def get_population(self):
        population = list()
        for _ in range(self.population_size):
            num_weights, num_biases = self.model.get_num_params()
            population.append([
                np.random.randn(num_weights),
                np.random.randn(num_biases)])

        return population

    def get_trial_weights(self, individual):
        trial_weights = list()
        for i, params in enumerate(individual):
            # make blur
            jitter = self.SIGMA * params
            # add to the weights (gaussian blurred weights)
            trial_weights.append(self.parameters[i] + jitter)

        return trial_weights

    def update_weights(self, rewards, population):
        update_rate = self.alpha / (self.population_size * self.SIGMA)
        
        # Normalise the reward
        rewards_std = rewards.std()
        if rewards_std == 0:
            return
        
        rewards = (rewards - rewards.mean()) / rewards_std
        for i, params in enumerate(self.parameters):
            # get the population for the given layer into an array
            w_try = np.array([p[i] for p in population])
            # print(w_try.shape)
            # print(params.shape)
            # update the weights
            # print((params + update_rate * np.dot(w_try.T, rewards).T).shape)
            self.parameters[i] = (params + update_rate * np.dot(w_try.T, rewards).T)
        # print(len(self.parameters[0]))
        # print(len(self.parameters[1]))
        # set the neural network weights with the updated parameters
        self.model.set_weights_biases(self.parameters[0], self.parameters[1])
        # update learning rate
        self.alpha *= 0.999

    def evolve(self):
        # make a worker pool to parallel compute the reward (for each individual)
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        for i in range(self.num_iterations):
            # get a population
            population = self.get_population()
            # compute the reward
            if pool is not None:
                # make a list of worker arguments
                worker_args = ((self.reward_function, self.get_trial_weights(individual))for individual in population)
                # run the pool
                rewards = pool.map(worker_process, worker_args)
                rewards = np.array(rewards)
            else:
                rewards = list()
                for individual in population:
                    trial_weights = self.get_trial_weights(individual)
                    rewards.append(self.reward_function(trial_weights))

                rewards = np.array(rewards)

            # Now update the weights of the ANN using the gradient update rule
            self.update_weights(rewards, population)

            if (i + 1) % self.print_step == 0:
                current_total_reward = self.reward_function(self.parameters)
                print('iter %d. reward: %f' % (i + 1, current_total_reward))
                if current_total_reward > self.save_point:
                    self.model.set_weights_biases(self.parameters[0], self.parameters[1])
                    self.model.save(self.save_path, i, current_total_reward)

        if pool is not None:
            pool.close()
            pool.join()


'''
_model = PolicyModel([4, 32, 32, 2])
nes = NaturalES(0.02, _model, 10)
pop = nes.get_population()
# print(len(pop[0][1]))
try_w = nes.get_trial_weights(pop[0])
# print(len(try_w[1]))

_rewards = np.random.randn(10)
nes.update_weights(_rewards, pop)
'''