import matplotlib.pyplot as plt
import numpy as np
import cv2

def aply_trashold(heatmap, trashold):
    heatmap[np.where(heatmap < trashold)] = 0.0
    return heatmap


def visualize_heatmap(heatmap, image=None, show_plot=False, fig_loc=None, trashold= None):

    if trashold is not None:
        heatmap= aply_trashold(heatmap, trashold)

    h, w, d = heatmap.shape[0], heatmap.shape[1], heatmap.shape[2]
    fig, ax = plt.subplots(nrows=1, ncols=d, figsize=[30, 5])

    for i, axi in enumerate(ax.flat):
        # axi is equivalent with ax[rowid][colid]
        image = cv2.resize(image, dsize=(heatmap.shape[0], heatmap.shape[1]), interpolation=cv2.INTER_NEAREST)
        axi.imshow(image, interpolation= 'none')
        axi.imshow(heatmap[:, :, i], "jet", interpolation= 'none', alpha=0.65)
        
    plt.tight_layout(pad=True)

    if fig_loc is not None:
        plt.savefig(fig_loc)
    
    if show_plot:
        plt.show()


def visualize_heatmap_with_ground_true(heatmap, ground_truth= None ,image=None, show_plot=False, fig_loc=None, trashold= None):

    if trashold is not None:
        heatmap= aply_trashold(heatmap, trashold)

    h, w, d = heatmap.shape[0], heatmap.shape[1], heatmap.shape[2]
    fig, ax = plt.subplots(nrows=2, ncols=d, figsize=[30, 10])

    for i, axi in enumerate(ax[0]):
        image = cv2.resize(image, dsize=(heatmap.shape[0], heatmap.shape[1]), interpolation=cv2.INTER_NEAREST)
        axi.imshow(image, interpolation= 'none')
        axi.imshow(heatmap[:, :, i], "jet", interpolation= 'none', alpha=0.65)
        axi.axis('off')
    
    if ground_truth is not None:
        for i, axi in enumerate(ax[1]):       
            image = cv2.resize(image, dsize=(ground_truth.shape[0], ground_truth.shape[1]), interpolation=cv2.INTER_NEAREST)
            axi.imshow(image, interpolation= 'none')
            axi.imshow(ground_truth[:, :, i], "jet", interpolation= 'none', alpha=0.65)
            axi.axis('off')
            
        
    plt.tight_layout()

    if fig_loc is not None:
        plt.savefig(fig_loc)
    
    if show_plot:
        plt.show()




if __name__ == "__main__":
    import tensorflow as tf
    from preprocess import Preprocessor

    p = Preprocessor(is_train=True, heatmap_shape=(128, 128, 8),)
    dataset = tf.data.Dataset.list_files('../img/hrany_rohy/train.tfrecords')
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(p)

    for image, heatmap in dataset.take(2):
        heatmap= heatmap.numpy()
        img= image.numpy()
        visualize_heatmap(heatmap,image=img, show_plot=True)
    