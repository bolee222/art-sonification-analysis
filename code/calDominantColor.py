from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorsys
from collections import Counter

# Load the image using mpimg.imread(). Use a raw string (prefix r) or escape the backslashes.

def calDominantColor(path, n_colors=16):
    image = mpimg.imread(path)
    
    # Get the dimensions (width, height, and depth) of the image
    w, h, d = tuple(image.shape)
    
    # Reshape the image into a 2D array, where each row represents a pixel
    pixel = np.reshape(image, (w * h, d))
    
    
    # Create a KMeans model with the specified number of clusters and fit it to the pixels
    model = KMeans(n_clusters=n_colors, random_state=42).fit(pixel)
    
    # Get the cluster centers (representing colors) from the model
    hue_list = [colorsys.rgb_to_hls(r,g,b) for [r,g,b,_] in model.cluster_centers_]
    colour_palette = np.uint8(model.cluster_centers_*255)

    labels = Counter(model.labels_)

    y = {
        'hue_list': hue_list,
        'centroids':colour_palette[:,:-1],
        'labels':labels}
    
    return y