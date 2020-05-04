# -*- coding: utf-8 -*-
"""
 @Time    : 19-11-21 下午4:27
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : train.py
"""
import multiprocessing

from keras.callbacks import TensorBoard, ReduceLROnPlateau, Callback
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from data_gen_label_cut import data_flow
from model.wasnet import WasterNet
from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler


def model_fn(FLAGS, objective, optimizer, metrics):
    model = WasterNet(input_shape = (FLAGS.input_size, FLAGS.input_size, 3), classes = FLAGS.num_classes)
    paralleled_model = multi_gpu_model(model, gpus = 3)
    paralleled_model.compile(loss = objective, optimizer = optimizer, metrics = metrics)
    return model, paralleled_model


class MyCbk(Callback):

    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs = None):
        self.model_to_save.save('./model_snapshots/model_at_epoch_%d.h5' % epoch)


def train_model(FLAGS):

    train_sequence, validation_sequence = data_flow(FLAGS.batch_size,FLAGS.num_classes, FLAGS.input_size)

    optimizer = RMSprop(lr = FLAGS.learning_rate)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy', 'top_k_categorical_accuracy']
    model, paralleled_model = model_fn(FLAGS, objective, optimizer, metrics)

    log_local = './log_file/Wasnet'
    tensorBoard = TensorBoard(log_dir = log_local)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, mode = 'auto')
    cbk = MyCbk(model)

    sample_count = len(train_sequence) * FLAGS.batch_size
    epochs = FLAGS.max_epochs
    warmup_epoch = 5
    batch_size = FLAGS.batch_size
    learning_rate_base = FLAGS.learning_rate
    total_steps = int(epochs * sample_count / batch_size)
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0,
                                            )
    paralleled_model.fit_generator(
        train_sequence,
        steps_per_epoch = len(train_sequence),
        epochs = FLAGS.max_epochs,
        verbose = 1,
        callbacks = [tensorBoard, reduce_lr, cbk],
        validation_data = validation_sequence,
        max_queue_size = 10,
        workers = int(multiprocessing.cpu_count( ) * 0.7),
        use_multiprocessing = True,
        shuffle = True
    )


def check_args(FLAGS):
    if not os.path.exists(FLAGS.train_url):
        os.mkdir(FLAGS.train_url)

def main(argv = None):
    check_args(FLAGS)
    train_model(FLAGS)

if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    import tensorflow as tf

    tf.app.flags.DEFINE_string('train_url', './model_snapshots/', 'the path to save training outputs')
    tf.app.flags.DEFINE_integer('num_classes', 40, 'the num of classes which your task should classify')
    tf.app.flags.DEFINE_integer('input_size', 224, 'the input image size of the model')
    tf.app.flags.DEFINE_integer('batch_size', 320, '')
    tf.app.flags.DEFINE_float('learning_rate', 1e-2, '')
    tf.app.flags.DEFINE_integer('max_epochs', 1000, '')

    FLAGS = tf.app.flags.FLAGS

    tf.app.run( )
