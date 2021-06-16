import open3d.ml.tf as ml3d
import open3d as o3d
import tensorflow as tf
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


path = '/media/2T/public_data/velodyne_sub/'

import yaml

def get_color_map(config_path):
    with open(config_path, 'rb') as fp:
        color_map = pickle.load(fp)
        print(color_map)

    return color_map

## get the all the labels and assign a color to eaqch
def assign_color_to_label(all_data):
    ## get all the unique values
    color_map = dict()
    for i in tqdm(range(len(all_data))):
        u_lbls = np.unique(all_data.get_data(i)['label'])
        for l in u_lbls:
            if l not in color_map.keys():
                color_map[str(l)] = list(np.random.uniform(0.0, 1.0, size=(3)))
            else:
                continue
    ## pickle dump
    print(color_map)
    
    with open('./semkitti_custom.p', 'wb') as fp:
        pickle.dump(color_map, fp, protocol=pickle.HIGHEST_PROTOCOL)

    ## check pickle
    with open('./semkitti_custom.p', 'rb') as fp:
        data = pickle.load(fp)
        print(data)

def convert_to_open3d(pcd, lbl, color_map_path):
    color_map = get_color_map(color_map_path)
    pcd_struct = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    
    point_colors = list(color_map[str(l)] for l in lbl)
    pcd_struct.colors = o3d.utility.Vector3dVector(point_colors)

    return pcd_struct

def data_generator(path, method, is_make_color_map=False):
    # construct a dataset by specifying dataset_path
    dataset = ml3d.datasets.SemanticKITTI(dataset_path=path)
    
    # get the 'all' split that combines training, validation and test set
    all_split = dataset.get_split(method)
    
    # print the attributes of the first datum
    #print(all_split.get_attr(0))

    if is_make_color_map:
        assign_color_to_label(all_split)
    
    num_sample = len(all_split)
    scaler = MinMaxScaler()

    def dataset_fn(sample_size):
        for i in range(num_sample):
            points = all_split.get_data(i)['point']

            labels = all_split.get_data(i)['label']
            ## downsample the point cloud using a uniform distribution
            _seed = np.int(all_split.path_list[i].split('/')[-1].split('.')[0])
            np.random.seed(_seed)
            
            u_idx = np.random.randint(0, points.shape[0], size=sample_size)
            points = points[u_idx]
            labels = labels[u_idx]
            
            yield points, labels 

    return dataset_fn, num_sample

def point_data_generator_tf(path, method, num_points):
    dataset, num_sample = data_generator(path, method)
    ## data generator
    pcd_generator = tf.data.Dataset.from_generator(dataset, 
                                                   args=[num_points], 
                                                   output_types=(tf.float32,
                                                                 tf.int32))
    
    return pcd_generator, num_sample

def one_hot_points(labels, num_classes):
    one_hot_batch = list()
    for label in labels:
        one_hot_batch.append(tf.one_hot(label.numpy(), num_classes))

    return tf.convert_to_tensor(one_hot_batch)

#################### debug only #####################################################
color_map_path = './semkitti_custom.p'

pcd_generator, num_sample = point_data_generator_tf(path, 'train', 5000)
#print(num_classes)
print(num_sample)

for pcd, label in pcd_generator.repeat().batch(1).take(10):
    print(pcd.shape)
    #print(label.shape)
    
    pcd = pcd.numpy()[0]
    lbl = label.numpy()[0]
    pcd_struct = convert_to_open3d(pcd, lbl, color_map_path)
    o3d.visualization.draw_geometries([pcd_struct])

'''
for pcd, label in pcd_generator.repeat().batch(1).take(10):
    print(pcd)
    print(one_hot_points(label, num_classes))
'''
