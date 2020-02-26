from __future__ import print_function, division

from constants import *
from oct_dataset import input_fn
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from my_densenet_nosobel import dense_net_model as densnet_nosobel
import matplotlib
matplotlib.use('Agg')


def save_history(history,const):
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    epochs_x = range(len(acc))
    plt.plot(epochs_x, acc, 'r', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.savefig(const.save_path + 'transfer_acc.png')
    plt.close()
    plt.plot(epochs_x, loss, 'r', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig(const.save_path + 'transfer_loss.png')
    plt.clf()
    plt.close()


def create_model_densnet_no_sobel(lr):
    model = densnet_nosobel(nclasses=2)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr,epsilon=0.1),
                  loss=tf.compat.v1.losses.softmax_cross_entropy,
                  metrics=['accuracy'])
    return model


def setup_training(const, train_fn=None, eval_fn=None, ls=False, nclasses=2,
                   batchsize=5, size=224):
    const.batch_size = batchsize
    const.epochs = 100

    train_data_fn = os.path.join(const.data_dir, train_fn)
    eval_data_fn = os.path.join(const.data_dir, eval_fn)
    train_data = pd.read_csv(train_data_fn, header=None, sep='\t')
    train_data.columns = ['fn', 'l']
    eval_data = pd.read_table(eval_data_fn, header=None, sep='\t')
    eval_data.columns = ['fn', 'l']

    train_data['fn'] = train_data['fn'].apply(lambda x: "{}{}".format(const.root_dir, x))
    eval_data['fn'] = eval_data['fn'].apply(lambda x: "{}{}".format(const.root_dir, x))

    train_inputs = input_fn(True, train_data['fn'].tolist(), train_data['l'].tolist(),
                            batch_size=const.batch_size, ls=ls,
                             nclasses=nclasses,size=size)
    eval_inputs = input_fn(False, eval_data['fn'].tolist(), eval_data['l'].tolist(),
                           batch_size=const.batch_size, ls=ls,
                            nclasses=nclasses,size=size)

    return eval_data_fn, train_inputs, eval_inputs, const
