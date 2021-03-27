import tensorflow as tf
import numpy as np
import random


class Preprocessor():
    def __init__(self,
                 image_shape=(256, 256, 3),
                 heatmap_shape=(64, 64, 16),
                 is_train=False):
        self.is_train = is_train
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape

    def __call__(self, example):
        features = self.parse_tfexample(example)
        image = tf.io.decode_jpeg(features['image/encoded'])
        image = tf.image.resize(image, self.image_shape[0:2])
        keypoints_x = features['image/corners/x']
        keypoints_y = features['image/corners/y']
        classes = features['image/corners/classes']

        if self.is_train:
            image= self.augmentate(image)

        image = tf.cast(image, tf.float32) / 127.5 - 1
        heatmaps = self.make_heatmaps(features, keypoints_x, keypoints_y, classes)
       
        return image, heatmaps

    def augmentate(self,image):        
        image= tf.image.random_contrast(image, 0.6 , 1.7, seed=None)    
        image= tf.image.random_brightness(image,20, seed=None)
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
        
        return image     


       
    def generate_2d_guassian(self,heatmap, height, width, y0, x0, sigma=1, scale=12):
        """
        Inšpirované

        https://github.com/princeton-vl/pose-hg-train/blob/master/src/util/img.lua#L204
        """

        # find bounds of gaussian
        xmin = x0 - 3 * sigma
        ymin = y0 - 3 * sigma
        xmax = x0 + 3 * sigma
        ymax = y0 + 3 * sigma

        #whole gaussian outside 
        if xmin >= width or ymin >= height or xmax < 0 or ymax <0 :
            return heatmap

        size = 6 * sigma + 1
        x, y = tf.meshgrid(tf.range(0, size, 1), tf.range(0, size, 1), indexing='xy')

        center_x = size // 2
        center_y = size // 2

        # generate this gaussian patch
        gaussian_patch = tf.cast(tf.math.exp(-(tf.square(x - center_x) + tf.math.square(y - center_y)) / (tf.math.square(sigma) * 2)) * scale, dtype=tf.float32)

        # correct if part of the patch is out of bounds
        patch_xmin = tf.math.maximum(0, -xmin)
        patch_ymin = tf.math.maximum(0, -ymin)
        patch_xmax = tf.math.minimum(xmax, width) - xmin
        patch_ymax = tf.math.minimum(ymax, height) - ymin

        # determine gausian position in heatmap
        heatmap_xmin = tf.math.maximum(0, xmin)
        heatmap_ymin = tf.math.maximum(0, ymin)
        heatmap_xmax = tf.math.minimum(xmax, width)
        heatmap_ymax = tf.math.minimum(ymax, height)

        # insert gaussian on it's position
        indices = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        
        count= 0
        for y in tf.range(patch_ymin, patch_ymax):
            for x in tf.range(patch_xmin, patch_xmax):
                indices = indices.write(count, [heatmap_ymin+y, heatmap_xmin+x])
                updates = updates.write(count, tf.math.maximum(gaussian_patch[y][x], heatmap[heatmap_ymin+y][heatmap_xmin+x]))
                count += 1

        return tf.tensor_scatter_nd_update(heatmap, indices.stack(), updates.stack())



    def make_heatmaps(self, features, keypoint_x, keypoint_y, classes):
        #relative to absolute coords 
        x = tf.cast(tf.math.round(keypoint_x * self.heatmap_shape[0]), dtype=tf.int32)
        y = tf.cast(tf.math.round(keypoint_y * self.heatmap_shape[1]), dtype=tf.int32)
        classes =  tf.cast(classes, dtype=tf.int32)
                
        #initialise heatmaps array
        heatmap_cnt = self.heatmap_shape[2]
        heatmap_array = tf.TensorArray(tf.float32, heatmap_cnt)
        for i in range(heatmap_cnt):
            heatmap_array.write(i, tf.zeros((self.heatmap_shape[0], self.heatmap_shape[1])))

        #initialise heatmap for each class
        cnt= tf.cast(features["image/corners/count"], dtype=tf.int32)
        for j in range(cnt):
            heatmap  = heatmap_array.read(classes[j]) #nacitanie heatmapy pre aktualnu triedu (pole indexovane triedami)
            heatmap = self.generate_2d_guassian(heatmap, self.heatmap_shape[1], self.heatmap_shape[0], y[j], x[j]) #pridanie bodu do heatmapy
            heatmap_array = heatmap_array.write(classes[j], heatmap) #zapis upravenej heatmapy do pola heatmap
        
        #stack into one heatmap with n channels where n is number of classes
        heatmaps = heatmap_array.stack()
        heatmaps = tf.transpose(heatmaps, perm=[1, 2, 0]) # reshape to (width, height, num_channels)
    
        return heatmaps

    def parse_tfexample(self, example_proto):
        image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/corners/count': tf.io.FixedLenFeature([], tf.int64),
            'image/corners/x': tf.io.RaggedFeature(tf.float32),
            'image/corners/y': tf.io.RaggedFeature(tf.float32),
            'image/corners/classes': tf.io.RaggedFeature(tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
           
        }
        features= tf.io.parse_single_example(example_proto, image_feature_description)

        return features
