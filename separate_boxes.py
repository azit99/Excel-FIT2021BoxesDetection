from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
mpl.rcParams['lines.linewidth'] = 2


def draw_rectangle(verts, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.add_patch(mpl.patches.Polygon(verts, closed=True, **kwargs))


def separate_boxes(heatmap, dwnsz= None):
    # vonkajsi obrys prahovany
    htm = heatmap[:, :, -1] > 1

    # izolovane komponenty

    labeled_array, num_features = ndimage.label(htm)
    MIN_SIZE=25
    for label in range(1, num_features+1):
        x, y = np.where((labeled_array == label).T)
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        if x_max-x_min < MIN_SIZE or y_max-y_min < MIN_SIZE:
            continue
      
        #draw_rectangle([[x_min*8, y_min*6], [x_min*8, y_max*6], [
        #               x_max*8, y_max*6], [x_max*8, y_min*6]], ax=None, fill=False)
        rect= [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]

        htm_sep = np.zeros_like(heatmap)
        htm_sep[y_min:y_max, x_min:x_max,
                :] = heatmap[y_min:y_max, x_min:x_max, :]

        yield htm_sep, rect


if __name__ == '__main__':
    from hourglass import StackedHourglassNetwork
    from visualizer import *

    # ziskanie heatmapy z modelu
    model = StackedHourglassNetwork(input_shape=(
        512, 512, 3), num_stack=2, num_residual=1, num_heatmap=8)
    model.load_weights('./MODEL_4.2.2/model-v0.0.1-epoch-37-loss-0.4046.h5')
    image = tf.io.decode_jpeg(tf.io.read_file('../img/hrany_rohy/0535.jpg'))
    inputs = tf.cast(tf.image.resize(image, (512, 512)),
                     tf.float32) / 127.5 - 1
    inputs = tf.expand_dims(inputs, 0)
    heatmap = model(inputs, training=True)[-1][0].numpy()
    dwnsz = image.numpy().astype(int)

    # skuska
    for htm, rect in separate_boxes(heatmap, dwnsz= dwnsz):
       
        plt.imshow(dwnsz)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        plt.imshow(htm[:, :, -1], "viridis",
                   interpolation='nearest', alpha=0.6, extent=(xmin,xmax,ymin,ymax))
    plt.show()
