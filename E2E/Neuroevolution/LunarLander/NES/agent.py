################################################### CODE IS BASED ON ######################################
# https://github.com/alirezamika/bipedal-es

from evolutionary_strategies import NaturalES
from model import PolicyModel
import numpy as np
import gym


#ACTION_MAP = {0:3, 1:4}
ACTION_MAP = None

class Agent:
    def __init__(self, env_name):
        self.SIGMA = 0.002
        self.policy_model = PolicyModel([8, 64, 64, 4])
        self.env = gym.make(env_name)
        
        population_size = 1000
        self.save_path = './np_log/lunar'
        '''
        gym.envs.register(
            id='CartPole-v2',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            max_episode_steps=3000,
            reward_threshold=950.0,
        )
        self.env = gym.make('CartPole-v2')
        '''
        self.nes = NaturalES(self.SIGMA,self.policy_model, population_size, self.save_path,
                                self.play_episode, alpha=0.001, num_threads=-1,
                                num_iterations=1000, save_point=250)

    def play_episode(self, trial_weights):
        # Make a model using the trial weights
        self.policy_model.set_weights_biases(trial_weights[0], trial_weights[1])
        # Run an episode
        current_obs = np.expand_dims(self.env.reset(), 0)
        done = False
        total_reward = 0.0
        while not done:
            action = np.argmax(self.policy_model(current_obs), -1)
            #print(action[0])
            # self.env.render()
            if ACTION_MAP:
                action = ACTION_MAP[action[0]]
            else:
                action = action[0]
            # exec the action on the env
            next_obs, reward, done, _ = self.env.step(action)
            current_obs = np.expand_dims(next_obs, 0)
            total_reward += reward

        return total_reward

    def train(self):
        self.nes.evolve()

    def test(self, model_name):
        # Load mode
        self.policy_model.load(self.save_path, model_name)
        for _ in range(5):
            current_obs = np.expand_dims(self.env.reset(), 0)
            done = False
            total_reward = 0.0
            while not done:
                action = np.argmax(self.policy_model(current_obs), -1)
                #print(action[0])
                self.env.render()
                if ACTION_MAP:
                    action = ACTION_MAP[action[0]]
                else:
                    action = action[0]
                # exec the action on the env
                next_obs, reward, done, _ = self.env.step(action)
                current_obs = np.expand_dims(next_obs, 0)
                total_reward += reward

            print("Episode Reward: ", total_reward)
            print()
        return total_reward


if __name__ == '__main__':
    env_name = 'LunarLander-v2'
    c_agent = Agent(env_name)
    #c_agent.train()
    c_agent.test('989_302_7282996.npy')