import math
import os
from datetime import datetime
import click
import tensorflow as tf
from hourglass import StackedHourglassNetwork
from preprocess import Preprocessor
import numpy as np


IMAGE_SHAPE = (512, 512, 3)
HEATMAP_SIZE = (128, 128)
MODEL_NAME= "MODEL_5"

class Trainer(object):
    def __init__(self,
                 model,
                 epochs,
                 global_batch_size,
                 strategy,
                 initial_learning_rate,
                 version='0.0.1',
                 start_epoch=1,
                 tensorboard_dir='./logs'):
        self.start_epoch = start_epoch
        self.model = model
        self.epochs = epochs
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        self.model = model

        self.current_learning_rate = initial_learning_rate
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf
        self.lowest_train_loss = math.inf
        self.patience_count = 0
        self.max_patience = 3
        self.tensorboard_dir = tensorboard_dir
        self.best_model = None
        self.version = version

    
    def compute_loss(self, labels, outputs):
        loss = 0.0
        for output in outputs:
            foreground_weights = tf.cast(labels > 0, dtype=tf.float32)
            background_weights = tf.cast(labels <= 0, dtype=tf.float32)
            foreground_cnt= tf.math.count_nonzero(foreground_weights)
            background_cnt= tf.math.count_nonzero(background_weights)
            def f1(): return foreground_cnt
            def f2(): return background_cnt
            foreground_cnt= tf.cond(tf.math.equal(foreground_cnt, 0), f2, f1)

            foreground_to_background= tf.cast(foreground_cnt / (foreground_cnt +background_cnt ), dtype=tf.float32)          

            foregorund_weights= foreground_weights * (1- foreground_to_background)
            background_wights= background_weights * foreground_to_background
            weights= foregorund_weights+background_weights          
            loss += tf.math.reduce_mean(tf.math.square(labels - output)*weights)
        return loss


    def train_step(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            loss = self.compute_loss(labels, outputs)
        grads = tape.gradient( target=loss, sources=self.model.trainable_variables ) 
        self.optimizer.apply_gradients( zip(grads, self.model.trainable_variables ))

        return loss

    def lr_decay(self):
        if self.patience_count == self.max_patience:
            self.current_learning_rate/= 10
            self.optimizer.learning_rate= self.current_learning_rate

    def val_step(self, inputs):
        images, labels = inputs
        outputs = self.model(images, training=True)
        loss = self.compute_loss(labels, outputs)
        return loss

    def run(self, train_dist_dataset, val_dist_dataset):
        @tf.function
        def distributed_train_epoch(dataset):
            tf.print('Start distributed traininng...')
            total_loss = 0.0
            num_train_batches = 0.0
            for one_batch in dataset:
                per_replica_loss = self.strategy.run(
                    self.train_step, args=(one_batch, ))
                batch_loss = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += batch_loss
                num_train_batches += 1
                tf.print('Trained batch', num_train_batches, 'batch loss',
                         batch_loss, 'epoch total loss', total_loss)
            return total_loss, num_train_batches

        @tf.function
        def distributed_val_epoch(dataset):
            total_loss = 0.0
            num_val_batches = 0.0
            for one_batch in dataset:
                per_replica_loss = self.strategy.run(self.val_step, args=(one_batch, ))
                num_val_batches += 1
                batch_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                tf.print('Validated batch', num_val_batches, 'batch loss',batch_loss)
                if not tf.math.is_nan(batch_loss):             
                    total_loss += batch_loss
                else:
                    num_val_batches-=1
        
            return total_loss, num_val_batches

        summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        summary_writer.set_as_default()

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.lr_decay()
            tf.summary.experimental.set_step(epoch)
            tf.summary.scalar('epoch learning rate', self.current_learning_rate)

            print('Start epoch {} with learning rate {}'.format( epoch, self.optimizer.learning_rate))

            train_total_loss, num_train_batches = distributed_train_epoch(train_dist_dataset)
            train_loss = train_total_loss / num_train_batches
            print('Epoch {} train loss {}'.format(epoch, train_loss))
            tf.summary.scalar('epoch train loss', train_loss)

            val_total_loss, num_val_batches = distributed_val_epoch(val_dist_dataset)
            val_loss = val_total_loss / num_val_batches
            print('Epoch {} val loss {}'.format(epoch, val_loss))
            tf.summary.scalar('epoch val loss', val_loss)

            #ukladanie najnizsiho train losu pre ucely lr shedulingu
            if train_loss < self.lowest_train_loss:
                self.lowest_train_loss = train_loss
                self.patience_count= 0
            else:
                self.patience_count+=1

            # ulozi model ak dosiahol novy najlepsi validation loss
            self.save_model(epoch, val_loss)
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss

        return self.best_model

    def save_model(self, epoch, loss):
        model_name = './'+MODEL_NAME+'/model-v{}-epoch-{}-loss-{:.4f}.h5'.format(
            self.version, epoch, loss)
        self.model.save_weights(model_name)
        self.best_model = model_name
        print("Model {} saved.".format(model_name))


def create_dataset(filenames, batch_size, num_heatmap, is_train):
    preprocess = Preprocessor(IMAGE_SHAPE, (HEATMAP_SIZE[0], HEATMAP_SIZE[1], num_heatmap), is_train)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
    if is_train:
        dataset = dataset.shuffle(batch_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def train(epochs, start_epoch, learning_rate, tensorboard_dir, checkpoint,
          num_heatmap, batch_size, train_tfrecords, val_tfrecords, version):
    strategy = tf.distribute.MirroredStrategy()
    global_batch_size = strategy.num_replicas_in_sync * batch_size
    train_dataset = create_dataset(train_tfrecords, global_batch_size, num_heatmap, is_train=True)
    val_dataset = create_dataset(val_tfrecords, global_batch_size, num_heatmap, is_train=False)

    if not os.path.exists(os.path.join('./'+MODEL_NAME)):
        os.makedirs(os.path.join('./'+MODEL_NAME))

    with strategy.scope():
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

        model = StackedHourglassNetwork(IMAGE_SHAPE, 2, 1, num_heatmap)
        print(model.summary())

        if checkpoint and os.path.exists(checkpoint):
            model.load_weights(checkpoint)

        trainer = Trainer(
            model,
            epochs,
            global_batch_size,
            strategy,
            initial_learning_rate=learning_rate,
            start_epoch=start_epoch,
            version=version,
            tensorboard_dir=tensorboard_dir)

        print('Start training...')
        return trainer.run(train_dist_dataset, val_dist_dataset)


if __name__ == "__main__":
    
    train_tfrecords = ['./dataset/train.tfrecords']
    val_tfrecords = ['./dataset/test.tfrecords']
    batch_size = 1
    num_heatmap = 8
    learning_rate = 0.001
    start_epoch = 0
    epochs=100
    tensorboard_dir= './'+MODEL_NAME
    #checkpoint= "./"+MODEL_NAME+"/model-v0.0.1-epoch-40-loss-0.3375.h5"
    checkpoint= None

    train(epochs, start_epoch, learning_rate, tensorboard_dir, checkpoint,
          num_heatmap, batch_size, train_tfrecords, val_tfrecords, '0.0.1')
