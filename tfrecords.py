import pandas as pd
import tensorflow as tf
import random
import math
from shutil import copyfile
import os

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  
        # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(features, content):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list = tf.train.Int64List(value=[features["heigth"]])),
        'image/width':tf.train.Feature(int64_list = tf.train.Int64List(value=[features["width"]])),
        'image/depth':tf.train.Feature(int64_list = tf.train.Int64List(value=[3])),
        'image/corners/count': tf.train.Feature(int64_list = tf.train.Int64List(value=[features["count"]])),
        'image/corners/x':tf.train.Feature(float_list = tf.train.FloatList(value=features["keypoints_x"])),
        'image/corners/y':tf.train.Feature(float_list = tf.train.FloatList(value=features["keypoints_y"])),
        'image/corners/classes':tf.train.Feature(int64_list = tf.train.Int64List(value=features["keypoint_classes"])),
        'image/filename': _bytes_feature(features["filename"].encode()),
        'image/encoded': _bytes_feature(content),       
    }))
    return tf_example

def remove_duplicates(data, size=128):
    df= data.copy()
    df['x'] = df['x'].apply(lambda x: round(x*size))
    df['y'] = df['y'].apply(lambda x: round(x*size))
    df = df.drop_duplicates(subset=['x', 'y'], keep='last')
    data= data.loc[df.index]
    return data




if __name__ == '__main__':
    classMap= {'1f':0, '1b':1, '2f':2, '2b':3, '3f':4, '3b':5, 'v':6, 'o':7} 
    #classMap= {'v':0, 'o':1}
    #classMap= {"3":2, "2":1, "1":0}
    DATASET_DIR="./dataset/"
    TRAIN_TFR_PATH = os.path.join(DATASET_DIR, "train.tfrecords")
    TEST_TFR_PATH = os.path.join(DATASET_DIR, "test.tfrecords")
    CSV_FILENAME= "labels_points.csv"
    TEST_PERCENT = 0.1
    VALID_PERCENT = 0.1
    TEST_PATH= os.path.join(DATASET_DIR, 'test')
    TRAIN_PATH= os.path.join(DATASET_DIR, 'train')
    VALID_PATH= os.path.join(DATASET_DIR, 'valid')

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(VALID_PATH):
        os.makedirs(VALID_PATH)
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    csv = pd.read_csv(os.path.join(DATASET_DIR, CSV_FILENAME))
    #csv= remove_duplicates(csv, 128) #remove points which will overlap in heatmap after coversion to absolute coords
    #csv = csv.replace({"2f": "2", "2b":"2", "1f":"1", "1b":"1", "3f":"3", "3b":"3"})
    filenames = csv.filename.unique()
    file_cnt= filenames.shape[0]
    print(math.ceil(TEST_PERCENT*file_cnt))
    test_indices = random.sample(range(1, file_cnt), math.ceil(TEST_PERCENT*file_cnt))
    rest = [x for x in range(1, file_cnt) if x not in test_indices]
    valid_indices = random.sample(rest, math.ceil(VALID_PERCENT*file_cnt))
    annot_cnt=0

    with tf.io.TFRecordWriter(TEST_TFR_PATH) as test_writer,  tf.io.TFRecordWriter(TRAIN_TFR_PATH) as train_writer :
        for filename in filenames :
            annot_cnt +=1
            filepath = DATASET_DIR+filename
            
            with open(filepath, 'rb') as image_file:
                image = image_file.read()

            width = csv.loc[csv.filename == filename]['width'].iloc[0]
            heigth = csv.loc[csv.filename == filename]['height'].iloc[0]
            features={
                "filename": filename,
                "width": width,
                "heigth": heigth,
                "keypoints_x": csv.loc[csv.filename == filename]['x'].values / width,
                "keypoints_y": csv.loc[csv.filename == filename]['y'].values / heigth,
                "keypoint_classes": [classMap[key] for key in csv.loc[csv.filename == filename]['class'].values],
            }
            features["count"] = features["keypoints_x"].shape[0]
            example = create_tf_example(features, image)
            print(filename)

            if annot_cnt in valid_indices:
                test_writer.write(example.SerializeToString())
                copyfile(os.path.join(DATASET_DIR, filename), os.path.join(VALID_PATH, filename))
            elif annot_cnt in test_indices:
                copyfile(os.path.join(DATASET_DIR, filename), os.path.join(TEST_PATH, filename))
            else:                
                train_writer.write(example.SerializeToString())
                copyfile(os.path.join(DATASET_DIR, filename), os.path.join(TRAIN_PATH, filename))

        print(annot_cnt, ' annotations written into',TEST_TFR_PATH, ' a ', TRAIN_TFR_PATH )
        