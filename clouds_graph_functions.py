""" script python qui contient toutes les fonctions utiles à l'affichage """

from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import imgaug.imgaug
import colorama
from colorama import Fore
import clouds_utilities_functions as cuf


def plot_image_with_masks(dataset, image_path):
    '''
    Function to visualize several segmentation maps of an image.
    INPUT:
        image_id - filename of the image
    '''
    # open the image
    image = np.asarray(Image.open(image_path))
    
    # draw segmentation maps and labels on image
    image = cuf.draw_segmentation_maps(dataset, image_path)
    
    # visualize the image and map
    side_by_side = np.hstack([image])
    
    labels = cuf.get_labels(dataset, image_path.split('/')[-1])

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axis('off')
    plt.title('Segmentation maps:' + labels)
    plt.legend()
    
    ax.imshow(side_by_side)


def plot_images_and_masks(dataset, images_path, width = 2, height = 3):
    """
    Function to plot grid of images and their labels with several examples of cloud images from train set.
    INPUT:
        width - number of images per row
        height - number of rows

    OUTPUT: None
    """
    # get a list of images
    images = sorted(glob(images_path + '*.jpg'))
    
    fig, axs = plt.subplots(height, width, figsize=(20, 20))
    
    # create a list of random indices 
    rnd_indices = [np.random.choice(range(0, len(images))) for i in range(height * width)]
    
    for im in range(0, height * width):
        # open image with a random index
        image = Image.open(images[rnd_indices[im]])
        # draw segmentation maps and labels on image
        image = cuf.draw_segmentation_maps(dataset, images[rnd_indices[im]])
        
        i = im // width
        j = im % width
        
        # plot the image
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')
        axs[i,j].set_title(cuf.get_labels(dataset, images[rnd_indices[im]].split('/')[-1]))

    # set suptitle
    plt.suptitle('Sample images from the train set')
    plt.show()


def visualize_image_mask_prediction(image, mask, mask_prediction, Transparency):
    """ Fonction pour visualiser l'image original, le mask original et le mask predit"""
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    if Transparency == True:       
        f, ax = plt.subplots(1, 5, figsize=(24,8))
        ax[0].imshow(image)
        ax[0].set_title('Image original', fontsize=fontsize)
    
        image = (image * 255).astype(np.uint8)
        for i in range(4):
            masko = mask[:, :, i]
            mask_pred = mask_prediction[:, :, i]
    
            segmap = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
            double_mask = masko + mask_pred
            segmap = np.where(masko == 1, 2, segmap) # 2: couleur verte
            segmap = np.where(mask_pred == 1, 3, segmap) #3: couleur jaune
            segmap = np.where(double_mask == 2, 4, segmap) #4: couleur bleue
            segmap = SegmentationMapOnImage(segmap,
                                            shape=image.shape,
                                            nb_classes=4,)
            imageMask = np.asarray(segmap.draw_on_image(image)).reshape(image.shape)
            ax[i + 1].imshow(imageMask)
            ax[i + 1].set_title(f'Comparaison masques {class_dict[i]}', fontsize=fontsize)
            
    else:
        f, ax = plt.subplots(2, 5, figsize=(24,8))

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Image original', fontsize=fontsize)
    
        for i in range(4):
            ax[0, i + 1].imshow(mask[:, :, i], vmin = 0, vmax = 1)
            ax[0, i + 1].set_title(f'Masque original {class_dict[i]}', fontsize=fontsize)
    
        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Image original', fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask_prediction[:, :, i],vmin = 0, vmax = 1)
            ax[1, i + 1].set_title(f'Masque predit {class_dict[i]}', fontsize=fontsize) 


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''
    
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('Vrais valeurs')
        plt.xlabel('Valeurs prédites' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()