import gym
import numpy as np

from model import PolicyModel


def play_episode(policy_model, env):
    
    # print(tf.reduce_max(current_obs))
    # print(tf.reduce_min(self.current_obs))

    reward_list = list()
    for i in range(1000):
        # episode loop
        current_obs = env.reset()
        # print(np.expand_dims(current_obs, 0).shape)
        current_obs = np.expand_dims(current_obs, 0)
        print("Run: ", i+1)
        while True:
            # take an action
            action = np.argmax(policy_model(current_obs), 1)
            env.render()
            #print(action)
            # action = tf.argmax(action_probs, 1)[0].numpy()
            # print(action)
            # action = ACTION_MAP[action]

            next_obs, reward, done, _ = env.step(action[0])
            next_obs = next_obs
            current_obs = np.expand_dims(next_obs, 0)

            reward_list.append(reward)

            if done:
                break
        # print(sum(reward_list))
    # return np.sum(reward_list), len(reward_list)  # np.random.uniform(21, -21, 1)[0]


# make the cartpole env
env = gym.make('CartPole-v0')
# Initialise A model
model = PolicyModel([4, 32, 32, 2])
model.load('./saved_elites', '2_1000_7176187.npy')

play_episode(model, env)