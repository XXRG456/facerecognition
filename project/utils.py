import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

def bgr_to_rgb(image:np.array):
    blue = image[:,:,0].copy()
    red = image[:,:,2].copy()
    image[:,:,2] = blue
    image[:,:,0] = red
    return image

def preprocess(image: np.array, cvt_color: bool):
    image = cv2.resize(image, (100, 100)) / 255.0
    image = image.astype("float32")
    if cvt_color:
        return bgr_to_rgb(image)
    return image

def create_anchor_images(anchors, faces):
    data = {}
    for key, value in anchors.items():
        data[key] = np.stack([preprocess(faces[key][pic], False) for pic in value])
    return data


def plot_anchor_images(anchor_images):

    fig, axes = plt.subplots(4, 7, figsize = (22,5))
    for row, (key, value) in enumerate(anchor_images.items()):
        index = np.random.randint(len(value), size = 7)
        for col, i in enumerate(index):
            axis = axes[row, col]
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.imshow(value[i])
        axes[row, 3].set_title(key, fontsize=12, pad=10)
    plt.tight_layout()
    
    
def create_prediction_pairs(anchors, prediction_face):
    data = {}
    for key in anchors.keys():
        placeholder = []
        for i in range(len(anchors[key])):
            anch = anchors[key][i]
            placeholder.append(np.array([anch, prediction_face]))
        placeholder = np.array(placeholder)    
        data[key] = placeholder   
    return data


def concat_pair(pair: np.array):
    return K.concatenate([pair[0], pair[1]], axis = 1)

def plot_pairs(prediction_pairs):
    
    fig, axes = plt.subplots(4, 7, figsize = (22,5))
    for row, (key, batch) in enumerate(prediction_pairs.items()):
        index = np.random.randint(len(batch), size = 7)
        for col, i in enumerate(index):
            axis = axes[row, col]
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.imshow(concat_pair(batch[i]))
        axes[row, 3].set_title(key, fontsize=12, pad=10)
    plt.tight_layout()