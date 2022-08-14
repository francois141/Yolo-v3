import torch
import cv2
import numpy as np
import os

# Simple bbxo_iou implementation
def bbox_iou(e1,e2):

    # Intersection area
    rect_x1 = torch.max(e1[:,0],e2[:,0])
    rect_x2 = torch.min(e1[:,2],e2[:,2])
    rect_y1 = torch.max(e1[:,1],e2[:,1])
    rect_y2 = torch.min(e1[:,3],e2[:,3])

    inter_area = torch.clamp(rect_x2 - rect_x1 + 1, min = 0) * torch.clamp(rect_y2 - rect_y1 + 1, min = 0)

    # Union area
    iou_area = (e1[:,2] - e1[:,0] + 1) * (e1[:,3] - e1[:,1] + 1) + (e2[:,2] - e2[:,0] + 1) * (e2[:,3] - e2[:,1] + 1) - inter_area

    # Return the result
    return inter_area / iou_area


def unique(tensor):
    return torch.unique(tensor)

def load_classes(path):
    return open(path,"r").read().split("\n")[:-1]

# Conver openCV format into image
def prep_image(image_resized, inp_dim):
    # Read the image and get the image
    image_origin = cv2.imread(image_resized)
    return prep_frame(image_origin,inp_dim)

def prep_frame(image_origin,inp_dim):
    dimension_origin = image_origin.shape[1],image_origin.shape[0]
    
    # Resize the image
    image_resized = (resize_image(image_origin, (inp_dim, inp_dim)))

    # Pass it into a tensor format
    image_resized = image_resized[:,:,::-1].transpose((2,0,1)).copy()

    # Pass the image into a tensor and rescale it
    image_resized = torch.tensor(image_resized) / 255.0

    # Convert the image into batch format
    image_resized = image_resized.unsqueeze(0)
    
    # Return the values
    return image_resized, image_origin, dimension_origin

# Resize and keep the ratio consitant
def resize_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''

    # Old dimensions
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim

    # Rescale the images
    min_ratio = min(inp_dim[0]/img.shape[1],inp_dim[1]/img.shape[0])
    new_w = int(img_w * min_ratio)
    new_h = int(img_h * min_ratio)

    # Resize the image
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    # Create a new image using the previous one
    new_image = np.full((inp_dim[1], inp_dim[0], 3), 128)
    new_image[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:] = resized_image
    
    # Return the new image 
    return new_image

def get_predictions(prediction,batch_size, confidence, num_classes,batch_offset,nms_conf = 0.4):
    
    # Take measures bigger than a threshold value
    prediction *= (prediction[:,:,4] > confidence).float().unsqueeze(2)

    # Pass predictions from top-left/right,width,height -> x1,y1,x2,y2
    height = torch.clone(prediction[:,:,2])
    width  = torch.clone(prediction[:,:,3])
    corner_x = torch.clone(prediction[:,:,0])
    corner_y = torch.clone(prediction[:,:,1])

    prediction[:,:,0] = corner_x - height / 2
    prediction[:,:,1] = corner_y - width / 2
    prediction[:,:,2] = corner_x + height / 2
    prediction[:,:,3] = corner_y + width / 2

    output = torch.zeros(size=(0,8))
    
    # we can do non max suppression only on in dividual images so we will loop through images
    for ind in range(batch_size): 

        # Get the image
        image_pred = prediction[ind] 

        # Get for each prediction the output with highest probability
        max_conf,max_conf_score = torch.max(image_pred[:,5: 5 + num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        # Concatenate rectangles with max score and index of max score
        image_pred = torch.cat((image_pred[:,:5], max_conf, max_conf_score),1)

        # Keep images that are are not masked out
        image_pred_ = image_pred[torch.nonzero(image_pred[:,4]).squeeze(),:].view(-1,7)

        try:
            # List all the detected classes 
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
       
        # Now we can apply non-max-suppression element wise
        for current_class in img_classes:

            # Get only images of the correct class
            class_mask = image_pred_* (image_pred_[:,-1] == current_class).float().unsqueeze(1)
            image_pred_class = image_pred_[torch.nonzero(class_mask[:,-2]).squeeze()].view(-1,7)
            
            # sort them based on probability
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )
            image_pred_class = image_pred_class[conf_sort_index[1]]
            idx = image_pred_class.size(0)
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                
                # Zero out all the detections that have IoU > treshhold
                image_pred_class[i+1:] *= (ious < nms_conf).float().unsqueeze(1)
                
                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_index = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_index, image_pred_class
            
            out = torch.cat(seq,1)
            output = torch.cat((output,out))
    output[:, 0] += batch_offset
    return output


def rescale_predictions(im_dimension_origin_list,output,inp_dim):
    # We need to rescale the predictions to the old image dimensions
    im_dimension_origin_list = torch.index_select(im_dimension_origin_list, 0, output[:,0].long())
    scaling_factor = torch.min(inp_dim/im_dimension_origin_list,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dimension_origin_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dimension_origin_list[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
            
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dimension_origin_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dimension_origin_list[i,1])

    return output

def write_rectangles(x, batches, results,classes):
    c1 = tuple(x[1:3].int().detach().cpu().numpy())
    c2 = tuple(x[3:5].int().detach().cpu().numpy())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (0,0,255)
    cv2.rectangle(img, c1, c2,color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


def output_images(im_dimension_origin_list,output,inp_dim,imlist,output_dir,im_batches,orig_ims,classes):

    # Rescale images
    output = rescale_predictions(im_dimension_origin_list=im_dimension_origin_list, output=output, inp_dim=inp_dim)

    # Write recatangles to the outputs
    list(map(lambda x: write_rectangles(x, im_batches, orig_ims, classes), output))

    # Prepare output_name
    det_names = ["{}/output_{}".format(output_dir,os.path.split(path)[1]) for path in imlist]

    # Save images
    list(map(cv2.imwrite, det_names, orig_ims))



def get_image_paths(input_dir):
    if not os.path.isdir:
        print("The input directory is not a directory")
        exit()
    return [os.path.join(input_dir,img) for img in os.listdir(input_dir)]