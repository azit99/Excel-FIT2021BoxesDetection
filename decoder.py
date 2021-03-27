import tensorflow as tf
from timeit import default_timer as timer
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preprocess import Preprocessor
import numpy as np
from hourglass import StackedHourglassNetwork
from visualizer import *
import numpy.matlib as npm
from scipy import ndimage
import math
from shapely.geometry import Polygon
import matplotlib as mpl
import itertools as iter
from separate_boxes import separate_boxes
from collections import namedtuple

#namedtuples definition
Corner = namedtuple('Corner', 'x y cls')
Edge = namedtuple("Edge", 'p1 p2 cls' )


def draw_rectangle(verts, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.add_patch(mpl.patches.Polygon(verts, closed=True, **kwargs))

def get_rectangle_from_cwha(cx_, cy_, w_, h_, a_):
    theta = math.radians(-a_)
    bbox = npm.repmat([[cx_], [cy_]], 1, 4) + \
        np.matmul([
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)]
        ],
        [
            [-w_/2,  w_/2,  w_/2,  -w_ / 2],
            [-h_ / 2, -h_/2, h_/2, h_ / 2]
        ])
    x1, y1 = bbox[0][0], bbox[1][0]
    x2, y2 = bbox[0][1], bbox[1][1]
    x3, y3 = bbox[0][2], bbox[1][2]
    x4, y4 = bbox[0][3], bbox[1][3]

    return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

def get_heatmap_centroids(htm, cls):
    centroids = []
    labeled_array, num_features = ndimage.label(htm)

    for label in range(1, num_features+1):
        x, y = np.where(labeled_array == label)
        p= Corner(np.mean(x), np.mean(y), cls)
        centroids.append(p) 
    return centroids


def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def is_between(a, c, b):
    return math.isclose(distance(a, c) + distance(c, b), distance(a, b), rel_tol=2e-4)

def corect_triangles(edges, cor):
    corners= [(p[0],p[1]) for p in cor]
    print("correcting")
    for (C1, C2, C3) in iter.combinations(corners, 3):
        cur_edg= [(C1, C2), (C2, C1), (C2, C3), (C3, C2), (C3, C1), (C1, C3)]
        if  ((C1, C2) in edges or (C2, C1) in edges)  and \
            ((C2, C3) in edges or (C3, C2) in edges) and \
            ((C3, C1) in edges or (C1, C3) in edges): 
            print("najdene")      
     
            edges= [e for e in edges if not(e[0] in (C1, C2, C3) and e[1] in (C1, C2, C3))]     
            rect, C= rec_from_3_points([C1, C2, C3])
            corners.append(C)
            edges+= rect

    return edges

def get_miss_edge(edges):
    new_edges=[]
    new_pair= []
    for e in edges:
        for p in e:
            found=False
            for e2 in edges:
                if e2 == e:
                    continue
                if e2[0] == p or e2[1] == p:
                    found= True
            if not found:
                new_pair.append(p)
        if len(new_pair) == 2:
            new_edges.append(new_pair)
            new_pair=[]
    return new_edges
                

def get_pairs(rohy_all,htm):
    # finding corner pairs which forms edges
    RECT_WIDTH = 5
    hrany = []

    edge_candidates= list(iter.combinations(rohy_all, 2))
    edge_candidates.sort(key=lambda x: distance(x[0], x[1]))
    for (P1, P2) in edge_candidates:
        if P1[:2] == P2[:2]:
            continue
        center = ((P1[0]+P2[0])/2, (P1[1]+P2[1])/2)
        angle = - \
            math.degrees(math.atan((P2[0] - P1[0]) / (P2[1] - P1[1] + 10e-10)))
        width = RECT_WIDTH
        height = distance(P1, P2)
        rect = get_rectangle_from_cwha(
            center[0], center[1], width, height, angle)

        mask = Image.new('L', (htm.shape[0], htm.shape[1]), 0)
        ImageDraw.Draw(mask).polygon(
            list(rect.exterior.coords), outline=1, fill=1)
        mask = np.array(mask)
        area = np.count_nonzero(mask)
        valid_area = np.count_nonzero(htm[np.where(mask == 1)])

        plt.title("podiel nenulovej plochy:"+str(valid_area / area))  
        rect_full_size= [(x*8,y*6) for x,y in rect.exterior.coords]      
       # draw_rectangle( rect_full_size, fill=False)         
       # plt.scatter(P1[0]*8, P1[1]*6)
       # plt.scatter(P2[0]*8, P2[1]*6)
       # plt.imshow(image)
       # xmin, xmax = plt.xlim()
       # ymin, ymax = plt.ylim()
       # plt.imshow(htm, cmap= "viridis", interpolation= "none", alpha=0.5, extent=(xmin,xmax,ymin,ymax))
       # plt.show()

        if(valid_area / area > 0.6):
            for x,y in hrany:
                if mask[int(x[0]), int(x[1])] == 1 and mask[int(y[0]), int(y[1])] ==1:
                    continue
            hrany.append((P1, P2))
            rect = get_rectangle_from_cwha(center[0], center[1], width-1, height-3, angle)
            mask = Image.new('L', (htm.shape[0], htm.shape[1]), 0)
            ImageDraw.Draw(mask).polygon(
                list(rect.exterior.coords), outline=1, fill=1)
            mask = np.array(mask)
            htm[np.where(mask == 1)]= 0

    return hrany

def rec_from_3_points(pts):
    import math 
    sides = [(pts[0],pts[1]), (pts[1], pts[2]), (pts[2], pts[0])]        
    sides= sorted(sides, key= lambda x : distance(*x))
    longest_side= sides[-1]
    del sides[-1]

    pts=[]
    for s in sides:
        pts.append(s[0])
        pts.append(s[1])

    for p in pts:
        if pts.count(p) == 2:
            A=p
            pts.remove(A)            
            pts.remove(A)

    B,C = pts
    vect = [(C[0] - A[0]) , (C[1] - A[1])]
    D=[B[0]+vect[0],B[1]+vect[1]]

    sides.append((D, B))
    sides.append((D, C))
    return sides, D

def remove_dupli_cornerns(corners):
    TRASHOLD= 6
    for c1 in corners:
        for c2 in corners:
            if c1 == c2:
                 continue
            if distance(c1, c2) <= TRASHOLD :
                corners.remove(c1)
    return corners

def cluster_box_edges(edges):
    def find_conection(edge, edges):
        for i in range(len(edges)):    
            if edge[0] in (edges[i][0], edges[i][1]) or edge[1] in (edges[i][0], edges[i][1]):
                return i
        return None

    boxes= []
    while len(edges) > 0:
        box= [edges[0]]
        del edges[0]
        while True:
            con_idx= find_conection(box[-1], edges)
            if con_idx is None:
                break
            else:
                box.append(edges[con_idx])
                del edges[con_idx]
        boxes.append(box)
    return boxes


if __name__ == "__main__":

    model = StackedHourglassNetwork(input_shape=(
        512, 512, 3), num_stack=2, num_residual=1, num_heatmap=8)
    model.load_weights('./MODEL5Sigma2/model-v0.0.1-epoch-47-loss-0.3214.h5')
    image = tf.io.decode_jpeg(tf.io.read_file('../img/hrany_rohy/0010.jpg'))
    inputs = tf.image.resize(image, (512, 512))
    inputs = tf.cast(inputs, tf.float32) / 127.5 - 1
    inputs = tf.expand_dims(inputs, 0)    
    outputs = model(inputs, training=True)[-1]
    heatmap_whole = outputs[0].numpy()
    heatmap_whole[:, :, :6] = heatmap_whole[:, :, :6] > 1.5
    image = image.numpy()

    for heatmap, rect in separate_boxes(heatmap_whole):        
        rohy=[]
        #ziskanie vsetkych rohov po triedach
        classes=["1f", "1b", "2f", "2b", "3f", "3b"]
        for i in range(6):
            rohy+=get_heatmap_centroids(heatmap[:, :, i].T, classes[i])
        rohy= remove_dupli_cornerns(rohy)
      
        map= heatmap[:, :, 6]+ heatmap[:, :, 7]
       
        htm = heatmap[:, :, 6]            
        htm[htm < 1] = 0          
        hrany_in= get_pairs(rohy, htm)
    
        htm = heatmap[:, :, 7]
        htm[htm < 1] = 0
        hrany_out= get_pairs(rohy, htm)
        
        hrany= get_miss_edge(hrany_out)
        hrany=hrany_in+hrany_out
        print(hrany)
        #boxes = cluster_box_edges(hrany)
        #hrany = boxes[0] #Pre testovanie iba jeden TODO
        hrany= corect_triangles(hrany, rohy)
        

        plt.imshow(image)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.imshow(map, cmap="viridis", interpolation="none", alpha=0.5, extent=(xmin,xmax,ymin,ymax))
        for roh in rohy:
            plt.scatter(roh[0]*8, roh[1]*6, color="white")
        for hrana in hrany:
            if hrana in hrany_in:
                color= "red"
            else:
                color= "black"
            x = [hrana[0][0]*8, hrana[1][0]*8]
            y = [hrana[0][1]*6, hrana[1][1]*6]
            plt.plot(x, y ,c=color)
        rect=[(x*8, y*6) for x,y in rect]
        draw_rectangle(rect, fill=False)
        plt.show()
