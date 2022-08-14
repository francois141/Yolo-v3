import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt

from yolo.Darknet import *
from yolo.utils import *

# Some variables
weightsfile = 'yolov3.weights'
classfile = 'coco.names'
cfgfile = 'yolov3.cfg'
sample_img1 = 'dog-cycle-car.png'
input_dir = 'input'
output_directory = 'output'
nms_thesh = 0.5
CUDA = False
batch_size = 1


# Make sure both folders exitst
if not os.path.exists(input_dir):
    os.mkdir(input_dir)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load the model
print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Loaded network with success.....")

# Load the classes
print("Loading the classes.....")
classes = load_classes(classfile)
print("Loaded classes with success.....")

# Some safety assertions
input_dimension = int(model.net_info["height"])
assert input_dimension % 32 == 0
assert input_dimension > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

# Get the path of the images
print("Loading the images.....")
images_path = get_image_paths(input_dir)
print("Loaded the images with success.....")


# Preparing the images
print("Rescaling the images.....")
batches = list(map(prep_image, images_path, [input_dimension for x in range(len(images_path))]))
print("Rescaled images with success.....")

# Resized, original image with the dimension list
images_resized = [x[0] for x in batches]
image_origin = [x[1] for x in batches]
image_base_dimensions = [x[2] for x in batches]
image_base_dimensions = torch.FloatTensor(
    image_base_dimensions).repeat(1, 2)

# If cuda is available send the image to cuda
if CUDA:
    image_base_dimensions = image_base_dimensions.cuda()


output = torch.zeros(size=(0,8))

for i,batch in enumerate(images_resized):
    # If cuda available put into cuda
    if CUDA:
        batch = batch.cuda()


    # Apply offsets to the result predictions
    # Tranform the predictions as described in the YOLO paper
    # flatten the prediction vector
    # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
    # Put every proposed box as a row.
    # Perform inference using the yolo model (Darknet 53)
    with torch.no_grad():
        predictions = model(batch, CUDA)

    # Get the predictions
    predictions = get_predictions(
        predictions, batch_size, confidence=0.5, num_classes=80, batch_offset=i*batch_size,nms_conf=nms_thesh)

    # Concatenate output
    output = torch.cat((output, predictions))

    if CUDA:
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detections were made")
    exit()


# Outputpipeline
output_images(image_base_dimensions,output,input_dimension,images_path,output_directory,images_resized,image_origin,classes)