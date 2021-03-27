import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
from hourglass import StackedHourglassNetwork
from preprocess import Preprocessor
import glob 
import pandas as pd
import numpy as np
from scipy import ndimage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import code

#enum
POINTS= 1
EDGES= 2

classMap= {'1f':0, '1b':1, '2f':2, '2b':3, '3f':4, '3b':5, 'v':6, 'o':7}   
#classMap= {'o': 1, "v":0}   
#classMap= {"3":2, "2":1, "1":0}
FOLDER= './dataset/test/'
CSV_PATH= './dataset/labels_points.csv'
POINT_TRASHOLD= 1.9
EDGE_TRASHOLD=4
ALPHA= 3

#model
IMG_INPUT_SHAPE= (512, 512, 3)
NUM_STACK= 2
NUM_HEATMAP= 8
MODEL_PATH= './MODEL_5'
OUTPUT_PATH= MODEL_PATH+'/test_data_eval.txt'
WEIGHTS_PATH= MODEL_PATH+ '/model-v0.0.1-epoch-50-loss-0.3529.h5'

###########################################################################################################################
def euclidian_dist(x, y, x1, y1):
    return math.sqrt((x-x1)**2  + (y-y1)**2)

def plot_grund_true(x,y, heatmap):
    fig,ax = plt.subplots(1)
    ax.imshow(heatmap, cmap='viridis', interpolation='nearest')
    for act_x, act_y in zip(x, y):
        plt.scatter(act_x , act_y, s=1, c='red', marker='o')

def predict_img(path, model):
    image = tf.io.decode_jpeg( tf.io.read_file(path))
    inputs = tf.image.resize(image, IMG_INPUT_SHAPE[:2])
    inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
    inputs = tf.expand_dims(inputs, 0)
    outputs = model(inputs, training=True)
    heatmap = outputs[-1].numpy()
    return heatmap

def get_heatmaps_centroids(htm):
    centroids= []
    labeled_array, num_features = ndimage.label(htm)

    for label in range(1, num_features+1):
        x, y = np.where(labeled_array == label)
        centroids.append((np.average(x, weights=htm[x,y]), np.average(y, weights=htm[x,y]))) 
    return np.asarray(centroids)


def evaluate_localization(heatmap, data, results, trashold, obj_type, VISUALIZE=False):
    htm = heatmap.sum(axis=2)
    htm[np.where(htm < trashold)]= 0
    
    if obj_type == POINTS :
        detections= get_heatmaps_centroids(htm)
    elif obj_type == EDGES:
        detections=np.argwhere(htm > 0) 

    #relative to absolute coords
    all_x= (data["x"].values / data["width"].values) * heatmap.shape[0]
    all_y= (data["y"].values / data["height"].values) * heatmap.shape[0]
    gt_points= np.dstack((all_x, all_y))[0]
    gt_points = gt_points.round()
    gt_points= np.unique(gt_points, axis=0)
    incorrect=[]
    detected_ground_true=[]

    for y, x in detections:
        distances= np.hypot(*(gt_points - [x,y]).T)
        if distances.min() < ALPHA:
            if gt_points[np.argmin(distances)].tolist() not in detected_ground_true:
                detected_ground_true.append(gt_points[np.argmin(distances)].tolist())
            results["correct_points"]+=1
        else:
            incorrect.append((x, y))
            results["incorrect_points"]+=1
    if VISUALIZE:
        cmap = plt.cm.viridis
        norm = plt.Normalize(htm.min(), htm.max())
        rgba = cmap(norm(htm))
    
        not_detected= [x for x in gt_points.tolist() if x not in detected_ground_true]
       #code.interact(local=locals())
        
        for (y, x) in detected_ground_true:
            rgba[int(x) , int(y), :3] = 0, 0, 1
        for (y, x) in np.around(incorrect):
            rgba[int(x) , int(y), :3] = 1, 0, 0
        for (y, x) in np.around(not_detected):
            rgba[int(x) , int(y), :3] = 1, 1, 1
        plt.imshow(rgba, interpolation='nearest')
        plt.show()
    
    results['detected_gt']+= len(detected_ground_true)
    results['total_gt']+= gt_points.shape[0]
    results["total_det"]+=detections.shape[0]
    return results


if __name__ == "__main__":

   # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    #nacitabie modelu
    model = StackedHourglassNetwork(input_shape=IMG_INPUT_SHAPE, num_stack=NUM_STACK, num_residual=1,num_heatmap=NUM_HEATMAP)
    model.load_weights(WEIGHTS_PATH)
    
    image_paths = glob.glob(os.path.join(FOLDER,'*.jpg'))[:20] #POCET 
    data = pd.read_csv(CSV_PATH)
    classes= np.unique(data["class"].values).tolist()
    results_hrany = {"correct_points":0 , "incorrect_points":0, "total_det":0, "total_gt":0, "detected_gt":0, 'perc_valid':0, 'perc_detected':0}
    results_rohy = {"correct_points":0 , "incorrect_points":0, "total_det":0, "total_gt":0, "detected_gt":0, 'perc_valid':0, 'perc_detected':0}


    for path in image_paths:
        heatmap = predict_img(path, model)
        if NUM_STACK > 1:
            heatmap = heatmap[0]
        
        filename= os.path.basename(path)
        print(filename)
        file_data = data.loc[data["filename"] == filename]


        results_corners = evaluate_localization(heatmap[:, :, :6], file_data[file_data['class'].isin(['1f', '1b', '2f', '2b', '3f', '3b'])],results_rohy, POINT_TRASHOLD, POINTS)
        results_edges = evaluate_localization(heatmap[:, :, 6:], file_data[file_data['class'].isin(['o', 'v'])],results_hrany, EDGE_TRASHOLD, EDGES)
    
    results_edges['perc_valid']=  results_edges["correct_points"]/ results_edges["total_det"]
    results_corners['perc_valid']=  results_corners["correct_points"]/ results_corners["total_det"]

    results_edges['perc_detected']=  results_edges["detected_gt"]/ results_edges["total_gt"]
    results_corners['perc_detected']=  results_corners["detected_gt"]/ results_corners["total_gt"]
  
    
    #Output
    print("##########################################################################################################################################################################")
    print(results_hrany, results_rohy)
    params= { 
        'trashold_rohy': POINT_TRASHOLD, 
        'trashold_hrany': EDGE_TRASHOLD, 
        'alpha':ALPHA,
        'model':WEIGHTS_PATH
        }
    hrany = json.dumps(results_hrany, indent=4)
    rohy= json.dumps(results_rohy, indent=4)
    with open(OUTPUT_PATH, 'a') as f:
        print ("edges localisation \n",hrany,"\n \n", file= f)
        print ("corners localisation  \n",rohy,"\n \n", file= f)
        print()
        print(params, file=f)
