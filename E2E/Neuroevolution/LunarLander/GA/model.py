from math import tanh
import os.path
import time
from typing import Tuple
import numpy as np
from numpy.core.defchararray import mod
from numpy.core.fromnumeric import shape


def softmax(x):
    e = np.exp(x)
    return e / e.sum()

def relu(x):
    x[x<0] = 0
    return x

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
    
class PolicyModel:
    def __init__(self, layer_shapes: list) -> None:
        # initialise the weights
        self.layer_shapes = layer_shapes
        self.num_layers = len(layer_shapes) - 1
        # make the model layers as a list
        # [[4, 32], [32, 32], [32, 2]]
        self.model_weights = [np.zeros((layer_shapes[i - 1], layer_shapes[i])) \
                              for i in range(1, len(layer_shapes))]
        # [32, 32, 2]
        self.model_biases = [np.zeros((layer_shapes[i])) \
                             for i in range(1, len(layer_shapes))]

        self.weight_shapes = [np.prod(w.shape) for w in self.model_weights]
        # print(self.weight_shapes)
        '''
        for w in self.model_biases:
            print(w.shape)
        '''
        w, b = self.get_weights_biases()
        self.flat_weights_size = len(w)
        self.flat_biases_size = len(w)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # call the model by multiplying inputs by weights and adding the bias
        for i in range(self.num_layers):
            x = np.tanh(np.matmul(x, self.model_weights[i]) + self.model_biases[i])
        '''
        x = sigmoid(np.matmul(x, self.model_weights[0]) + self.model_biases[0])
        for i in range(1, self.num_layers-2):
            # print(x)
            x = relu(np.matmul(x, self.model_weights[i]) + self.model_biases[i])
        # x -> l1 -> x -> l2 -> x -> o
        x = softmax(np.matmul(x, self.model_weights[-1]) + self.model_biases[-1])
        '''
        return x

    def build(self, x: np.ndarray) -> np.ndarray:
        out = self(x)
        print(out.shape)
        return out

    def get_weights_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        # make a flat vector of weights and biases and return them for mutation
        # all_weights = [p for i in range(self.num_layers) for p in list(self.model_weights[i].flatten())]
        all_weights = np.concatenate([w.flatten() for w in self.model_weights], -1)
        all_biases = np.concatenate([b.flatten() for b in self.model_biases], -1)
        # print(type(all_weights))
        return all_weights, all_biases

    def set_weights_biases(self, new_weights: np.ndarray, new_biases: np.ndarray) -> None:
        start_w = 0
        start_b = 0
        # iterate for num layers, reshape new weights and set the model
        model_new_layer_weights = list()
        model_new_layer_biases = list()
        for i in range(self.num_layers):
            # print(self.weight_shapes[i])
            new_layer_weights = new_weights[start_w:self.weight_shapes[i] + start_w]
            # print(new_layer_weights.shape)
            new_layer_weights_reshaped = np.reshape(new_layer_weights, (self.layer_shapes[i], self.layer_shapes[i + 1]))
            new_layer_biases = new_biases[start_b:self.layer_shapes[i + 1] + start_b]

            start_w = self.weight_shapes[i]
            start_b = self.layer_shapes[i + 1]
            # print(start)
            # print(new_layer_weights_reshaped)
            # print(new_layer_biases)
            model_new_layer_weights.append(new_layer_weights_reshaped)
            model_new_layer_biases.append(new_layer_biases)

        # update the model
        self.model_weights = model_new_layer_weights
        self.model_biases = model_new_layer_biases

    def init_weights(self, type, seeds: int, init_bias=False) -> None:
        np.random.seed(seeds)
        if type == 'standard':
            new_w = np.random.normal(loc=0.0, scale=1.0, size=self.flat_weights_size) 
            if init_bias == True:
                new_b = np.random.normal(loc=0.0, scale=1.0, size=self.flat_biases_size) 
            else:
                new_b = np.zeros(shape=self.flat_biases_size)
            self.set_weights_biases(new_w, new_b)
        else:
            raise NotImplementedError("Method not implemented")

    def save(self, save_path: str, generation_id: int, reward: float) -> None:
        # make a layer dict
        layer_params = [{'layer': i, 'w': w, 'b': b}
                        for i, (w, b) in enumerate(zip(self.model_weights, self.model_biases))]
        # print(layer_params)
        random_string = str(time.time()).split('.')[-1]
        model_path = '_'.join([str(generation_id), str(int(np.floor(reward))), random_string]) + '.npy'
        with open(os.path.join(save_path, model_path), 'wb') as f:
            np.save(f, layer_params)

    def load(self, load_path: str, file_name: str) -> None:
        with open(os.path.join(load_path, file_name), 'rb') as f:
            layer_params = np.load(f, allow_pickle=True)
            # print(layer_params)
            loaded_weights = list()
            loaded_biases = list()
            for i, item in enumerate(layer_params):
                # print(item)
                w = item['w']
                b = item['b']
                loaded_weights.append(w)
                loaded_biases.append(b)
                # print(w.shape)
                # print(b.shape)
            # print(i)
            self.model_weights = loaded_weights
            self.model_biases = loaded_biases


'''
model = PolicyModel([4, 32, 32, 2])

init_w, init_b = model.get_weights_biases()
# print(init_w)
# print(init_b)
model.init_weights('standard', 1234)
init_w, init_b = model.get_weights_biases()
# print(init_w)
# print(init_b)
#model.save('./saved_elites', 1, 50.934343)
model.load('./saved_elites', '1_132_613585.npy')
# new_w, new_b = model.get_weights_biases()
model.build(np.random.uniform(-0.1, 0.1, size=(1,4)))
'''
# print(np.array_equal(init_w, new_w))
# print(np.array_equal(init_b, new_b))
# print(new_w.shape)
# print(new_b.shape)
# print(init_w.shape)
# print(init_b.shape)

'''
#model.get_weights_biases()
new_w = np.random.normal(0.0, 1.0, size=1216)
new_b = np.random.normal(0.0, 1.0, size=(32+32+2))

old_w, old_b = model.get_weights_biases()
#print(len(old_w))
#print(len(old_b))
model.set_weights_biases(new_w, new_b)
new_w, new_b = model.get_weights_biases()
#print(new_w)
#print(new_b)
#print(old_w - new_w)
print(np.array_equal(old_w, new_w))
print(np.array_equal(old_b, new_b))
'''
'''
for i in range(100):
    x = np.random.uniform(-100.0, 100.0, size=(1,4))
    preds = model(x)
    print(preds.shape)
    out = np.argmax(preds, 1)
    print(out)
'''