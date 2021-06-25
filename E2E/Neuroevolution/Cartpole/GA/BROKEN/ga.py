import numpy as np

import gym

from collections import deque

import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mean
from tensorflow.python.ops.numpy_ops.np_math_ops import mod
from tensorflow.python.ops.sort_ops import sort
import tensorflow_probability as tfp

from lr_schedular import CosineAnnealingSchedule
'''
def process_obs(obs):
    obs = 
'''
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
'''
ACTION_MAP = {0 : 3, 1 : 4}
## a simple model
def build_model(min_val, max_val, seed, input_shape=(210,160,3), resize_shape=(86,86)):
    #initializer = tf.keras.initializers.RandomUniform(min_val, max_val, seed=seed)
    initializer = tf.keras.initializers.RandomNormal(0.0, 0.01, seed=seed)
    inputs = tf.keras.layers.Input(input_shape)
    #normalized_inputs = tf.keras.layers.experimental.preprocessing.Normalization(
    #                                                                axis=-1)(inputs)
    resized_inputs = tf.keras.layers.experimental.preprocessing.Resizing(
        resize_shape[0], resize_shape[1], interpolation='bilinear'
    )(inputs)
    #kernel_initializer=initializer
    conv_1 = tf.keras.layers.Conv2D(32, 3,
                                activation='relu',
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                )(resized_inputs)
    pool_1 = tf.keras.layers.MaxPool2D()(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, 3,
                                activation='relu',
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                )(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D()(conv_2)                                
    flat = tf.keras.layers.Flatten()(pool_2)
    hidden = tf.keras.layers.Dense(512,
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,
                                    )(flat)
    action_logits = tf.keras.layers.Dense(2,
                                        activation='relu',
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer,
                                        )(hidden)
    actions = tf.keras.layers.Softmax()(action_logits)

    policy = tf.keras.Model(inputs=inputs, outputs=actions)
    
    policy.build(input_shape)

    return policy
'''
#model = build_model(1000)
#inp = tf.random.uniform((1,86,86,3), 0.0, 1.0)
#out = policy(inp)
#print(model.weights[4].shape)
#print(model.weights[5].shape)
#print(model.summary())
'''
class PolicyModel(tf.keras.Model):
    def __init__(self, seed):
        super(PolicyModel, self).__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, 
                                                        stddev=1.0, 
                                                        seed=seed)
    
        self.conv_1 = tf.keras.layers.Conv2D(32, 3, kernel_initializer=initializer)
        self.conv_2 = tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer)
        self.flat = tf.keras.layers.Flatten()
        self.actions = tf.keras.layers.Dense(2, kernel_initializer=initializer)

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.flat(x)

        action_logits = self.actions(x)

        return action_logits

inp = tf.random.uniform((1,86,86,3), 0.0, 1.0)
policy = PolicyModel(1000)
print(len(policy.weights))
out = policy(inp)
print(len(policy.weights))
#print(out.shape)
'''
'''
seed = 1000
initializer = tf.keras.initializers.RandomNormal(mean=0.0, 
                                                        stddev=1.0, 
                                                        seed=seed)

model = tf.keras.Sequential(
        [tf.keras.layers.Input((86,86,3)),
        tf.keras.layers.Conv2D(32, 3, kernel_initializer=initializer),
        tf.keras.layers.Conv2D(64, 3, kernel_initializer=initializer),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, kernel_initializer=initializer),]
)#.build()
model.build()
#print(model.weights)
'''

def build_model(seed, num_actions, input_shape):
    initializer = tf.keras.initializers.GlorotNormal(seed)
    #initializer = tf.keras.initializers.RandomNormal(0.0, 0.1, seed=seed)
    inputs = tf.keras.layers.Input(input_shape)
    normalised_inputs = tf.keras.layers.experimental.preprocessing.Normalization(
                                                                    axis=-1)(inputs)
    #resized_inputs = tf.keras.layers.experimental.preprocessing.Resizing(
    #    86, 86, interpolation='bilinear'
    #)(inputs)
    
    #kernel_initializer=initializer
    '''
    conv_1 = tf.keras.layers.Conv2D(32, 3
                                )(inputs)
    conv_2 = tf.keras.layers.Conv2D(64, 3)(conv_1)
    '''
    #flat = tf.keras.layers.Flatten()(inputs)
    hidden_1 = tf.keras.layers.Dense(32,
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,)(normalised_inputs)
    hidden_2 = tf.keras.layers.Dense(32,
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    bias_initializer=initializer,)(hidden_1)
    action_logits = tf.keras.layers.Dense(num_actions,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer,)(hidden_2)
    actions = tf.keras.layers.Softmax()(action_logits)

    policy = tf.keras.Model(inputs=inputs, outputs=actions)
    
    policy.build(input_shape)

    return policy
## class to make the GA 
class GeneticAlgorithm():
    def __init__(self, population_size, T, sigma, env,
                    num_generations) -> None:
        ## Initialise a set of neural networks
        self.generation_id = 0
        self.sigma = sigma ## mutation strength
        self.population_size = population_size

        #self.min_array = np.random.uniform(-0.00001, -0.01, population_size)
        #self.max_array = np.random.uniform(0.00001, 0.01, population_size)
        self.seed = np.random.randint(0, 10_000, population_size)
        #print(self.seed[0])
        self.model_population = deque(maxlen=population_size)
        self.elite_reward = 0
        self.T = T
        self.num_steps = 1
        
        self.env = env
        self.num_actions = env.action_space.n
        self.num_inputs = env.observation_space.shape

        self.scheduler = CosineAnnealingSchedule(0.0, sigma, num_generations)
        
    def evolve(self):
        self.sigma = self.scheduler(self.generation_id)
        self.generation_id += 1

        if self.generation_id == 1:
            for p in range(self.population_size):
                self.model_population.append({
                                            'model': build_model(self.seed[p],
                                                                self.num_actions,
                                                                self.num_inputs),
                                            'seed': self.seed[p],
                                            'reward': -21,
                                            })
        else:
            ## get a new mutated population
            ## before mutating make a copy of the deque as a list
            model_population_list = list(self.model_population)
            ## get the best individuals mean weights
            best_mean_weights = [tf.reduce_mean(p).numpy() \
                                for p in model_population_list[1]['model'].weights]
            ## clear the deque for new population
            self.model_population.clear()
            ## set the deque witht the new population
            for _ in range(1, self.population_size):
                k = np.random.randint(0, self.T)
                selected_individual = model_population_list[k]
                mutated_individual = self.mutate_individual(selected_individual,
                                                            best_mean_weights)
                ## clear the old population and make space for the new population
                self.model_population.append(mutated_individual)
            
            ## put back the elite
            self.model_population.insert(0, model_population_list[0])

        #print(self.model_population)
        print("Population size", len(self.model_population))
        population_mean_fitness, elite_reward = self.compute_fitness()

        return population_mean_fitness, elite_reward, self.sigma

    def mutate_individual(self, selected_individual, best_mean_weights):
        ## get the model weights and biases
        selected_model = selected_individual['model']
        current_seed = selected_individual['seed']
        current_reward = selected_individual['reward']

        weight_mag = 0
        '''
        for param in selected_model.weights:
            param_sum = tf.reduce_sum(tf.abs(param))
            weight_mag += param_sum
        '''
        #param_mean = [tf.reduce_mean(p).numpy() for p in selected_model.weights]
        
        new_seed = self.next_seed(current_seed, np.random.randint(-1000, 1000,
                                                                dtype=np.int))
        #print()
        #print("Seed", new_seed)
        #print()
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
        ## make a dict for storage
        mutated_individual = {
                                'model': selected_model,
                                'seed': new_seed,
                                'reward': current_reward
                            }
        
        return mutated_individual

        
    def play_episode(self, model):
        ## episode loop
        self.current_obs = self.env.reset()
        #print(self.current_obs.shape)
        self.current_obs = tf.expand_dims(
                                    tf.convert_to_tensor(self.current_obs),
                                    axis=0
                                )
        #print(tf.reduce_max(self.current_obs))
        #print(tf.reduce_min(self.current_obs))

        reward_list = list()
        
        while True:
            ## take an action
            action_probs = model(self.current_obs)
            dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
            action = np.int(dist.sample().numpy()[0])
            self.env.render()
            #print(action)
            #action = tf.argmax(action_probs, 1)[0].numpy()
            #print(action)
            #action = ACTION_MAP[action]

            next_obs, reward, done, _ = self.env.step(action)
            next_obs = next_obs
            self.current_obs = tf.expand_dims(
                                    tf.convert_to_tensor(next_obs),
                                    axis=0
                                )
            
            reward_list.append(reward)
            
            if done:
                break
        #print(sum(reward_list))
        return np.sum(reward_list), len(reward_list)#np.random.uniform(21, -21, 1)[0]

    def compute_fitness(self):
        
        population_fitness = list()
        start = 0 if self.generation_id == 1 else 1
        print(start)
        for i in range(start, len(self.model_population)):
            
            ## evaluate the model
            print("Gen, ", self.generation_id)
            print("Indi", i)
            print("Current fitness: ", self.model_population[i]['reward'])
            total_reward, self.num_steps = self.play_episode(
                                                self.model_population[i]['model'])
            self.model_population[i]['reward'] = total_reward
            #print("New mean fitness: ", self.model_population[i]['reward'])
            print("New fitness: ", self.model_population[i]['reward'])
            print("Num steps: ", self.num_steps)
            print()
            #print(total_reward)
            population_fitness.append(total_reward)

        mean_population_reward = np.mean(population_fitness)
        elite_reward = self.model_population[0]['reward']
        print(mean_population_reward)
        return mean_population_reward, elite_reward

    def next_seed(self, current_seed, parent_sum):
        next_seed = current_seed + parent_sum
        #print(sorted_index)
        return next_seed
    
    def sort_population(self):
        new_elite = False
        rewards = [r['reward'] for r in self.model_population]
        print("Reward ", rewards)
        sorted_index = np.argsort(rewards)[::-1]
        print(sort(rewards, direction='DESCENDING'))
        print("Sorted idx ", sorted_index)
        
        elite_model = self.model_population[0]
        model_population_list = list(self.model_population)
        print("indi 0 reward", model_population_list[0]['reward'])
        self.model_population.clear()
        if self.generation_id == 1:
            best_T = sorted_index[:5]
            print(best_T[0])
            self.elite_reward = rewards[best_T[0]]
            elite_model = model_population_list[best_T[0]]
        else:
            best_T = sorted_index[:4]
            if self.elite_reward < rewards[best_T[0]]:
                print("Elite reward, ", self.elite_reward)
                print("New reward, ", rewards[best_T[0]])
                self.elite_reward = rewards[best_T[0]]
                elite_model = model_population_list[best_T[0]]
                new_elite = True

            '''
            for idx in best_T:
                if self.elite_reward < rewards[idx]:
                    self.elite_reward = rewards[idx]
                    elite_model = self.model_population[idx]
                else:
                    elite_model = self.model_population[0]
            '''
        
        for i in range(1, len(model_population_list)):
            self.model_population.append(model_population_list[sorted_index[i]])

        self.model_population.insert(0, elite_model)
        '''
        if new_elite:
            self.model_population.pop()
            self.model_population.insert(self.T - 1, elite_model)
        ''' 

        print("Population size", len(self.model_population))
        print([r['reward'] for r in self.model_population])


def main(population_size, T, sigma, input_size):
    env = gym.make('CartPole-v0')
    num_generations = 1000
    ga_optimiser = GeneticAlgorithm(
                            population_size, 
                            T, 
                            sigma, 
                            env, num_generations)

    writer = tf.summary.create_file_writer('./logs/cartpole')
    with writer.as_default():
        for ep in range(num_generations):
            print()
            print("##### NEW POPULATION #####")
            print()
            population_reward, elite_reward, sigma_step = ga_optimiser.evolve()
            tf.summary.scalar('population mean reward', population_reward, ep+1)
            tf.summary.scalar('Elite reward', elite_reward, ep+1)
            tf.summary.scalar('sigma decay', sigma_step, ep+1)
            print()
            print("Population Mean: ", population_reward)
            print()
            ga_optimiser.sort_population()        
    

if __name__ == "__main__":
    main(1000, 20, 0.002, (1, 86,86, 3))
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
