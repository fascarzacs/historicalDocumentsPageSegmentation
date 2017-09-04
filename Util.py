import cv2
import numpy as np
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from matplotlib import pyplot as plt
import matplotlib.path as mplPath
import xml.etree.ElementTree as ET

def plotImage(img, size):
    fig, ax = plt.subplots(figsize=(size,size))
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img)
    plt.xticks([]),
    plt.yticks([])
    plt.show()
    
def subplot(titles, images, rows, imgPerRows, size):
    fig, ax = plt.subplots(figsize=(size,size))
    for i in range(len(images)):
        plt.subplot(rows,imgPerRows,i+1),
        #plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),
        plt.yticks([])
    plt.show()
    
def readPageImagesAndGroundTruth (folderPageImages, folderGroundTruth, subFolderGroundTruth) :
    listImages = []; listGroundTruth = []
    for fileGroundTruth in os.listdir(folderGroundTruth + "/" + subFolderGroundTruth) :
        if fileGroundTruth.endswith('.xml') :
            for file in os.listdir(folderPageImages) :
                if file.endswith('.jpg') and fileGroundTruth.find(file[0:5]) > -1 :
                    #image = cv2.imread(folderPageImages + "/" + file)
                    image = img_as_float(io.imread(folderPageImages + "/" + file))
                    listImages.append(image)
                    listGroundTruth.append(fileGroundTruth)
    return listImages, listGroundTruth

def resizeImages (images, factor):
    resizedImages = []
    for i in range (len(images)) :
        dim = (int(images[i].shape[1]*factor),int(images[1].shape[0]*factor))
        resizedImages.append(cv2.resize(images[i], dim, interpolation = cv2.INTER_AREA))
    return resizedImages

def groundThruthFindCountourPointsByRegion (pathGroundThruthFile, region) :
    listPixels = []
    polygonsByRegion = []
    pointsPolygon = []
    tree = ET.parse(pathGroundThruthFile)
    root = tree.getroot()

    for neighbor in root.iter('TextRegion') :
        if (neighbor.attrib['type'] == region) :
            pointsPolygon = []
            for coord in neighbor.findall('Coords') :
                for point in coord.findall('Point') :
                    pointsPolygon.append([point.attrib['x'], point.attrib['y']])
                polygonsByRegion.append(pointsPolygon)
                
    return polygonsByRegion

def isInsidePolygon (listPointsPolygon, x, y) :
    bbPath = mplPath.Path(listPointsPolygon)
    point = ([x, y])
    return bbPath.contains_point(point)

def paintPolygon (listPointPolygonRegion, listPointsInsidePolygon, image, factor, B, G, R) :
    imageWidth = image.shape[1] #Get image width
    imageHeight = image.shape[0] #Get image height
    xPos, yPos = 0, 0
    while xPos < imageWidth: #Loop through rows
        while yPos < imageHeight: #Loop through collumns
            if isInsidePolygon (listPointPolygonRegion, xPos*factor, yPos*factor) :
                image.itemset((yPos, xPos, 0), B)
                image.itemset((yPos, xPos, 1), G)
                image.itemset((yPos, xPos, 2), R)
                listPointsInsidePolygon.append([xPos,yPos])
            yPos = yPos + 1 #Increment Y position by 1

        yPos = 0
        xPos = xPos + 1 #Increment X position by 1    
        
def paintPointsOutsideList (listPoints, image, B, G, R) :
    imageWidth = image.shape[1] #Get image width
    imageHeight = image.shape[0] #Get image height
    #from IPython.core.debugger import Tracer; Tracer()() 
    xPos, yPos = 0, 0
    for i in range(len(listPoints)) :
        point = listPoints[i]
        x = point[0]
        y = point[1]
        while xPos < imageWidth: #Loop through rows
            while yPos < imageHeight: #Loop through collumns
                if xPos == x and yPos == y :
                    image.itemset((yPos, xPos, 0), B)
                    image.itemset((yPos, xPos, 1), G)
                    image.itemset((yPos, xPos, 2), R)

                yPos = yPos + 1 #Increment Y position by 1

            yPos = 0
            xPos = xPos + 1 #Increment X position by 1
            
def segmentImageInSuperpixels (images, numSuperpixels) :
    listSegmentsByImage = []
    for i in range(len(images)) :
        segments = slic(images[i], n_segments = numSuperpixels, sigma = 5)
        listSegmentsByImage.append(segments)
    return listSegmentsByImage
