import os
from hourglass import StackedHourglassNetwork
from preprocess import Preprocessor
import glob 
import tensorflow as tf
from separate_boxes import *

def get_iou(bb1, bb2):
    """
    https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    #intersection over union = intersection
    # area divided  by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == "__main__":
    IMG_FOLDER= '../img/hrany_rohy/test/'
    ANNOT_FOLDER='../img/hrany_rohy/test/bbox/'
    MODEL_PATH= './MODEL_4.2.2'
    OUTPUT_PATH= MODEL_PATH+'/test_data_eval.txt'
    WEIGHTS_PATH= MODEL_PATH+ '/model-v0.0.1-epoch-37-loss-0.4046.h5'

    #Results
    TP=0
    FP=0
    FN=0

    #vyhodnotenie
    model = StackedHourglassNetwork(input_shape=(512, 512,3), num_stack=2, num_residual=1,num_heatmap=8)
    model.load_weights(WEIGHTS_PATH)    
    image_paths = glob.glob(os.path.join(IMG_FOLDER,'*.jpg')) #POCET 

    for img_path in image_paths:
         #read gt annotation
        annot_file= os.path.basename(img_path)[:-4]+'.txt'
        f = open(ANNOT_FOLDER+annot_file, "r")
        print(annot_file)
        gt_bboxes= []
        for line in f:
            data = [float(x)*128 for x in line.split()]
            #from yolo to left bottom rigth top
            gt_bboxes.append({"x1":data[1]-data[3]/2, "y1":data[2]-data[4]/2, 
            "x2": data[1]+data[3]/2, "y2":data[2]+data[4]/2})
        
        #get predictions
        image = tf.io.decode_jpeg( tf.io.read_file(img_path))
        inputs = tf.image.resize(image, (512, 512))
        inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
        inputs = tf.expand_dims(inputs, 0)
        heatmap = model(inputs, training=True)[-1][-1].numpy()
        pred_boxes= [{"x1":rect[0][0], "y1":rect[0][1],"x2":rect[2][0], "y2":rect[2][1]} 
        for _, rect in separate_boxes(heatmap)]

        detected= []
        for box in pred_boxes:
            for gt in gt_bboxes:
                iou= get_iou(box,gt)
                if(iou > 0.5):
                    TP+=1
                    if gt not in detected:
                        detected.append(gt)
                else:
                    FP+=1
        FN+= len(gt_bboxes) - len(detected)

    precision= TP/(TP+ FP)
    recall= TP/ (TP+FN)
    print("precision:", precision)
    print("recall:", recall)

       







    

    