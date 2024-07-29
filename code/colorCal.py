import cv2
import os
from os import listdir
import numpy as np
from colorthief import ColorThief
import matplotlib.pyplot as plt


#  HSV the ranges are as H[0-179], S[0-255], V[0-255]

def calSaturation(img):
    #img = cv2.imread('../paintings2/' + filename + '.jpg', cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = img_hsv[:, :, 1].mean() /255 *100
    #print("Saturation of " + img  + ".jpg is :     " + saturation)
    return saturation
    


def reportSaturation():
    folder_dir = "../paintings2"  
    for image in os.listdir(folder_dir):
        if (image.endswith(".jpg")):
            img = cv2.imread('../paintings2/' + image, cv2.IMREAD_COLOR)
            imageName = os.path.basename(image)
            print ("Saturation of " + imageName  + " is :     " + str(calSaturation(img)))



def calHUE(imageName):
    color_thief = ColorThief('../paintings2/' + imageName)
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=6, quality=1)
    
    plt.figure(figsize=(10,2.3))
    circle1=plt.Circle((0,1.0), radius=1, edgecolor='k', facecolor=np.array(dominant_color) / 255, linewidth=1 )

    palette1=plt.Circle((2.0,1.0), radius=0.4, edgecolor='k', facecolor=np.array(palette[0]) / 255, linewidth=1 )
    palette2=plt.Circle((3.0,1.0), radius=0.4, edgecolor='k', facecolor=np.array(palette[1]) / 255, linewidth=1 )
    palette3=plt.Circle((4.0,1.0), radius=0.4, edgecolor='k', facecolor=np.array(palette[2]) / 255, linewidth=1 )
    palette4=plt.Circle((5.0,1.0), radius=0.4, edgecolor='k', facecolor=np.array(palette[3]) / 255, linewidth=1 )
    palette5=plt.Circle((6.0,1.0), radius=0.4, edgecolor='k', facecolor=np.array(palette[4]) / 255, linewidth=1 )
    palette6=plt.Circle((7.0,1.0), radius=0.4, edgecolor='k', facecolor=np.array(palette[5]) / 255, linewidth=1 )

    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(circle1)
    ax.add_patch(palette1)
    ax.add_patch(palette2)
    ax.add_patch(palette3)
    ax.add_patch(palette4)
    ax.add_patch(palette5)
    ax.add_patch(palette6)
    plt.xlim([-2,8])
    plt.ylim([-0.1,2.1])
    plt.axis('off')
    plt.savefig('../paintings2/palettes/' + imageName,dpi=600)
    
    print ("-------------------------------------")
    print ("Dominant Color of " + imageName  + " is :     " + str(dominant_color))
    print ("Palettes Colors of " + imageName  + " are :     " + str(palette))


def reportHUE():
    folder_dir = "../paintings2"  
    for image in os.listdir(folder_dir):
        if (image.endswith(".jpg")):
            imageName = os.path.basename(image)
            calHUE(imageName)


reportHUE()
