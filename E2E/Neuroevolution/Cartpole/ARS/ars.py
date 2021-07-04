import gym
import numpy as np
from model import PolicyModel


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

    def explore(self, is_render):
        explore_reward = self.play_episode((self.flat_w, self.flat_b), is_render)

        return explore_reward

    def train(self):
        # iterate for number of episodes
        for ep in range(self.num_iterations):
            positive_reward = list()
            negative_reward = list()
            # iterate for number of delta directions (samples)
            episode_deltas = list()

            for d in range(self.num_directions):
                # sample a delta
                deltas = self.sample_deltas()
                # print(len(deltas))
                episode_deltas.append(deltas)
                p_w, p_b, n_w, n_b = self.update_directions(deltas)
                # play episode and get the reward
                positive_reward.append(self.play_episode((p_w, p_b)))
                # play episode and get reward
                negative_reward.append(self.play_episode((n_w, n_b)))

            all_rewards = np.array(positive_reward + negative_reward)
            reward_sigma = all_rewards.std()

            # sort rollouts wrt max(r_pos, r_neg) and take (hp.b) best
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_reward, negative_reward))}
            order = sorted(scores.keys(), key=lambda x: scores[x])[-self.num_best:]
            rollouts = [(positive_reward[k], negative_reward[k], episode_deltas[k]) for k in order[::-1]]
            # update the model parameters
            self.update_model(rollouts, reward_sigma)

            if ep % self.log_f == 0:
                print(ep)
                total_reward = 0.0
                is_render = False
                if ep > 90:
                    is_render = True
                for _ in range(5):
                    total_reward += self.explore(is_render=is_render)

                print("Episode: {}, Test reward: {}".format(ep, total_reward / 5.0))


# dummy test the algorithm
_env = gym.make('LunarLander-v2')
ars = ARS(10, 100, 5, [32], 0.015, 5, _env)
# _deltas = ars.sample_deltas()
# print(len(_deltas))
# w_p, b_p, w_n, b_n = ars.update_directions(_deltas)
# print(w_p.shape)
# print(b_p.shape)
# print(w_n.shape)
# print(b_n.shape)
ars.train()
