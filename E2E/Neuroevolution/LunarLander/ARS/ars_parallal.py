# motivated examples #
# https://github.com/alexis-jacq/numpy_ARS.git
# https://github.com/iamsuvhro/Augmented-Random-Search.git

import gym
import numpy as np
from model import PolicyModel
import multiprocessing as mp


def worker_process(args):
    func, arg = args

    return func(arg)


def normalise_reward(rewards):
    r_mean = rewards.mean()
    r_std = rewards.std()
    if r_std == 0.0:
        return
    return (rewards - r_mean) / r_std


class ARS:
    def __init__(self, num_directions, num_iterations, num_best, layer_hidden_shapes, alpha, log_frequency, env):
        self.num_directions = num_directions
        self.num_iterations = num_iterations
        self.num_best = num_best
        self.alpha = alpha
        self.log_f = log_frequency

        self.env = env
        num_actions = [env.action_space.n]
        num_inputs = [env.observation_space.shape[0]]
        self.layer_shapes = num_inputs + layer_hidden_shapes + num_actions
        self.model = PolicyModel(self.layer_shapes)
        self.model.init_weights('standard', 1234)
        self.flat_w, self.flat_b = self.model.get_weights_biases()
        self.weight_shape = self.flat_w.shape[0]
        # print(self.weight_shape)
        self.bias_shape = self.flat_b.shape[0]
        # print(self.bias_shape)
        self.num_layers = len(self.layer_shapes)

    def sample_deltas(self):
        delta_weights = np.random.randn(self.weight_shape)
        delta_biases = np.random.randn(self.bias_shape)
        # print(delta_biases)
        # print(delta_weights)
        return (delta_weights, delta_biases)

    def play_episode(self, params, is_render=False):
        # set the parameters
        self.model.set_weights_biases(params[0], params[1])
        # reset the env
        obs = np.expand_dims(self.env.reset(), 0)
        done = False
        reward_list = list()
        while not done:
            # play episode
            if is_render:
                self.env.render()
            action = np.argmax(self.model(obs), -1)
            next_obs, reward, done, _ = self.env.step(action[0])
            next_obs = next_obs
            obs = np.expand_dims(next_obs, 0)

            reward_list.append(reward)

        episode_reward = np.sum(reward_list)

        return episode_reward

    def update_directions(self, deltas):
        # get the model current parameters
        # w, b = self.model.get_weights_biases()

        new_p_w = self.flat_w + deltas[0]
        new_p_b = self.flat_b + deltas[1]

        new_n_w = self.flat_w - deltas[0]
        new_n_b = self.flat_b - deltas[1]

        return new_p_w, new_p_b, new_n_w, new_n_b

    def update_model(self, rollouts, reward_sigma):
        step_w = np.zeros(self.flat_w.shape)
        step_b = np.zeros(self.flat_b.shape)

        for r_pos, r_neg, d in rollouts:
            step_w += (r_pos - r_neg)*d[0]
            step_b += (r_pos - r_neg)*d[1]
        self.flat_w += self.alpha * step_w / (reward_sigma*self.num_best)
        self.flat_b += self.alpha * step_b / (reward_sigma*self.num_best)

    def evaluate(self, is_render):
        explore_reward = self.play_episode((self.flat_w, self.flat_b), is_render)

        return explore_reward

    def explore(self, deltas):
        p_w, p_b, n_w, n_b = self.update_directions(deltas)
        # play positive
        positive_reward = self.play_episode((p_w, p_b))
        # play negative
        negative_reward = self.play_episode((n_w, n_b))

        return [positive_reward, negative_reward]

    def train(self):
        num_threads = -1
        num_cpus = mp.cpu_count() if num_threads == -1 else 1
        print(num_cpus)
        pool = mp.Pool(num_cpus) if num_cpus > 1 else None

        # pool_2 = mp.Pool(num_cpus // 2) if num_cpus > 1 else None
        # iterate for number of episodes

        for ep in range(self.num_iterations):
            # positive_reward = list()
            # negative_reward = list()
            # iterate for number of delta directions (samples)
            # episode_deltas = list()

            episode_deltas = [self.sample_deltas() for _ in range(self.num_directions)]
            # print(len(delta_list))
            worker_args = [(self.explore, deltas) for deltas in episode_deltas]
            # print(len(worker_args))
            both_rewards = np.array(pool.map(worker_process, worker_args))
            # print(np.array(both_rewards))
            positive_reward = both_rewards[:, 0]
            negative_reward = both_rewards[:, 1]
            # print(negative_reward)
            # positive_reward.append(p_r)
            # negative_reward.append(n_r)

            all_rewards = np.array(positive_reward + negative_reward)
            all_rewards = normalise_reward(all_rewards)
            reward_sigma = all_rewards.std()

            # sort rollouts wrt max(r_pos, r_neg) and take (hp.b) best
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_reward, negative_reward))}
            order = sorted(scores.keys(), key=lambda x: scores[x])[-self.num_best:]
            # print(order)
            # get the descending oder and get the corresponding positive, negative rewards and deltas
            rollouts = [(positive_reward[k], negative_reward[k], episode_deltas[k]) for k in order[::-1]]
            # update the model parameters
            self.update_model(rollouts, reward_sigma)

            if ep % self.log_f == 0:
                print(ep)
                total_reward = 0.0
                is_render = False
                if ep > 900:
                    is_render = True
                for _ in range(30):
                    total_reward += self.evaluate(is_render=is_render)

                print("Episode: {}, Test reward: {}".format(ep, total_reward / 30.0))

        if pool is not None:
            pool.close()
            pool.join()


# dummy test the algorithm
def main():
    _env = gym.make('LunarLander-v2')
    ars = ARS(100, 1000, 30, [32], 0.015, 5, _env)
    ars.train()


if __name__ == "__main__":
    main()

'''
_env = gym.make('LunarLander-v2')
ars = ARS(60, 100, 20, [32], 0.015, 5, _env)
# _deltas = ars.sample_deltas()
# print(len(_deltas))
# w_p, b_p, w_n, b_n = ars.update_directions(_deltas)
# print(w_p.shape)
# print(b_p.shape)
# print(w_n.shape)
# print(b_n.shape)
ars.train(pool)
'''