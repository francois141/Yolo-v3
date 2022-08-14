# Import the dependencies
import cv2
import pandas as pd
import torch

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

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    ret, frame = vid.read()

    # Resize the frame
    image_resized,image_origin,dimension_origin = prep_frame(frame,input_dimension)

    # Perform the inference
    predictions = model(image_resized,False)

    # Get the predictions
    predictions = get_predictions(predictions, 1, confidence=0.5, num_classes=80, batch_offset=0,nms_conf=nms_thesh)

    # Rescale the predictions
    predictions = rescale_predictions(torch.FloatTensor(dimension_origin).repeat(1, 2), output=predictions, inp_dim=input_dimension)

    # For each detected object : write a rectangle inside of the image
    for e in predictions:
        write_rectangles(e,[image_resized],[image_origin],classes)

    # Display the image
    cv2.imshow('frame', image_origin)
      
    # Keep running until we press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

# Release the video and destroy all windows
vid.release()
cv2.destroyAllWindows()