import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
import open3d as o3d
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import point_data_generator_tf, get_color_map, one_hot_points, \
convert_to_open3d
tf.random.set_seed(1234)
cfg_path = './semkitti_custom.p'
#data_path = '/media/FourT/public_data/data/velodyne'
NUM_POINTS = 5_000
NUM_CLASSES = 20# len(get_color_map(cfg_path))
BATCH_SIZE = 10
EPOCHS = 1000

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, 3))
x = tnet(inputs, 3)
x = conv_bn(x, 64)
x = conv_bn(x, 64)
x = tnet(x, 64)
x_64 = conv_bn(x, 64)
x = conv_bn(x, 128)
x = conv_bn(x_64, 256)
x = conv_bn(x, 1024)
global_pool = layers.GlobalMaxPooling1D()(x)

## stack pool n times and cat with x_64
multiples = [1, NUM_POINTS, 1]
global_x = tf.tile(tf.expand_dims(global_pool, 1), multiples)
x = layers.Concatenate(2)([x_64, global_x])
x = conv_bn(x, 512)
x = conv_bn(x, 256)
x = conv_bn(x, 128)
x = conv_bn(x, 128)

seg_logit = conv_bn(x, NUM_CLASSES)
seg_out = layers.Softmax()(seg_logit)

model = keras.Model(inputs=inputs, outputs=seg_out)
ckpt_dir = './checkpoints_ae'
load_pretrained = False 
if load_pretrained:
    print("Loading model...")
    ckpt = './checkpoints_ae/pcd_seg_model_sk_scale_20'
    model.load_weights(ckpt)

print(model.summary())
pcd_generator_train, num_sample = point_data_generator_tf(data_path, 'train', NUM_POINTS)
pcd_generator_train = pcd_generator_train.shuffle(BATCH_SIZE*4).repeat().batch(BATCH_SIZE)

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

def train_step(model, pcd, label):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = model(pcd, training=True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        _loss = loss_fn(label, logits)
        reg = tf.keras.losses.kullback_leibler_divergence(label, logits)
        loss_value = _loss + reg
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return loss_value

def normalise_points(points_batch):
    min_p = tf.reduce_min(points_batch, 1)
    max_p = tf.reduce_max(points_batch, 1)

    min_p_ex = tf.expand_dims(min_p, 1)
    max_p_ex = tf.expand_dims(max_p, 1)

    return tf.math.divide((points_batch - min_p_ex), (max_p_ex - min_p_ex)), min_p_ex, max_p_ex

mean_metric = tf.keras.metrics.Mean()
num_batches = num_sample // BATCH_SIZE
def train():
    for e in range(EPOCHS):
        mean_metric.reset_states()
        with tqdm(total=num_batches * BATCH_SIZE) as progress_bar:
            for pcd, label in pcd_generator_train.take(num_batches):
                pcd, _, _ = normalise_points(pcd)

                one_hot_label_batch = one_hot_points(label, NUM_CLASSES)

                loss = train_step(model, pcd, one_hot_label_batch)
                mean_metric.update_state(loss)
                #print(loss)
                progress_bar.update(BATCH_SIZE)

        print("Epoch: {}, Mean Loss: {}".format(e,
                                               mean_metric.result().numpy()))

        ## save model at the end of epoch
        model.save_weights('./checkpoints_seg_small_2/pcd_seg_model_normed_' + str(e+1))

def test():
    ckpt = './checkpoints_seg_semkitti_with_pooling/pcd_seg_model_normed_31'
    model.load_weights(ckpt)

    pcd_generator_test, num_sample = point_data_generator_tf(data_path, 'test', NUM_POINTS)
    for pcd, label in pcd_generator_test.take(5):
        pcd = tf.expand_dims(pcd, 0)
        pcd, min_p, max_p = normalise_points(pcd)
        label = label.numpy()
        pred_labels = model(pcd)
        pred_labels = tf.math.argmax(pred_labels, 2).numpy()[0]
        
        max_min = max_p - min_p
        pcd_denorm = pcd * max_min + min_p
        
        pred_o3d = convert_to_open3d(tf.squeeze(pcd_denorm, 0).numpy(), pred_labels, cfg_path)
        inp_o3d = convert_to_open3d(tf.squeeze(pcd_denorm, 0).numpy(), label, cfg_path)
        o3d.visualization.draw_geometries([inp_o3d])
        o3d.visualization.draw_geometries([pred_o3d])

#test()
train()
