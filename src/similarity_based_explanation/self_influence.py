# @title Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.image as mpimg
import io
import json
import numpy as np
import time
import datetime
import functools
import matplotlib.pyplot as plt
import sys
import torch
from torchvision.models import resnet50
from torch.utils.data import DataLoader


#@title Self Influence Utils

def show_self_influence(trackin_self_influence, topk=50):
    self_influence_scores = trackin_self_influence['self_influences']
    indices = np.argsort(-self_influence_scores)
    for i, index in enumerate(indices[:topk]):
        print('example {} (index: {})'.format(i, index))
        print('label: {}, prob: {}, predicted_label: {}'.format(
            index_to_classname[str(trackin_self_influence['labels'][index])][1], 
            trackin_self_influence['probs'][index][0], 
            index_to_classname[str(trackin_self_influence['predicted_labels'][index][0])][1]))
        img = get_image(trackin_self_influence['image_ids'][index])
        if img is not None:
            plt.imshow(img, interpolation='nearest')
            plt.show()


@tf.function
def run_self_influence(inputs):
    imageids, images, labels = inputs
    self_influences = []
    for m in models:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(m.trainable_weights[-2:])
            probs = m(images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        grads = tape.jacobian(loss, m.trainable_weights[-2:])
        scores = tf.add_n([tf.math.reduce_sum(
            grad * grad, axis=tf.range(1, tf.rank(grad), 1)) 
            for grad in grads])
        self_influences.append(scores)  

    # Using probs from last checkpoint
    probs, predicted_labels = tf.math.top_k(probs, k=1)
    return imageids, tf.math.reduce_sum(tf.stack(self_influences, axis=-1), axis=-1), labels, probs, predicted_labels


def run_self_influence(inputs):
    imageids, images, labels = inputs
    self_influences = []
    for m in models:
        m.eval()
        with torch.no_grad():
            probs = m(images)
            loss = torch.nn.functional.cross_entropy(probs, labels)
        grads = tape.jacobian(loss, m.trainable_weights[-2:])
        scores = tf.add_n([tf.math.reduce_sum(
            grad * grad, axis=tf.range(1, tf.rank(grad), 1)) 
            for grad in grads])
        self_influences.append(scores)  

    # Using probs from last checkpoint
    probs, predicted_labels = tf.math.top_k(probs, k=1)
    return imageids, tf.math.reduce_sum(tf.stack(self_influences, axis=-1), axis=-1), labels, probs, predicted_labels


def get_self_influence(ds):
    image_ids_np = []
    self_influences_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []
    for d in ds:
        imageids_replicas, self_influences_replica, labels_replica, probs_replica, predictied_labels_replica = strategy.run(run_self_influence, args=(d,))  
        for imageids, self_influences, labels, probs, predicted_labels in zip(
            strategy.experimental_local_results(imageids_replicas), 
            strategy.experimental_local_results(self_influences_replica), 
            strategy.experimental_local_results(labels_replica), 
            strategy.experimental_local_results(probs_replica), 
            strategy.experimental_local_results(predictied_labels_replica)):
        if imageids.shape[0] == 0:
            continue
        image_ids_np.append(imageids.numpy())
        self_influences_np.append(self_influences.numpy())
        labels_np.append(labels.numpy())
        probs_np.append(probs.numpy())
        predicted_labels_np.append(predicted_labels.numpy()) 
    return {'image_ids': np.concatenate(image_ids_np),
            'self_influences': np.concatenate(self_influences_np),
            'labels': np.concatenate(labels_np),
            'probs': np.concatenate(probs_np),
            'predicted_labels': np.concatenate(predicted_labels_np)
            }    

if __name__ == "__main__":
    start = time.time()
    with strategy.scope():
        models = []
        for i in [30, 60, 90]:
            model = resnet.resnet50(1000)
            model.load_weights(CHECKPOINTS_PATH_FORMAT.format(i))
            models.append(model)
    end = time.time()
    print(datetime.timedelta(seconds=end - start))    
    
    ds_train = strategy.experimental_distribute_datasets_from_function(make_get_dataset('train', 512))
    
    start = time.time()
    trackin_train_self_influences = get_self_influence(ds_train)
    end = time.time()
    print(datetime.timedelta(seconds=end - start))

    show_self_influence(trackin_train_self_influences, topk=300)