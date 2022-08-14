
import numpy as np
import torch
import torch.nn as nn


def predictions(x,inp_dim,anchors,num_classes, CUDA = False):

    # Get batch & grid size
    batch_size = x.size(0)
    grid_size = x.size(2)

    # Get the stride
    stride =  inp_dim // x.size(2)

    # Number of bounding boxes per attribute & the number of anchors
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # Resize the predictions
    prediction = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Divide the anchors by the stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Transform coordinates
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])

    # Apply sigmoid on the object score
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Create the offset
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1) #(1,gridsize*gridsize,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    # Add the offset to the predictions
    prediction[:,:,:2] += x_y_offset

    # Repeat the anchors to apply them element-wise 
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)

    # Resize the values using the anchors as well as the stride
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    prediction[:,:,:4] *= stride    

    # Finally we can sigmoid the number of classes
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes])) 

    return prediction


def parse_cfg(config_file):
    file = open(config_file,'r')
    file = file.read().split('\n')
    file =  [line for line in file if len(line)>0 and line[0] != '#']
    file = [line.lstrip().rstrip() for line in file]

    final_list = []
    element_dict = {}
    for line in file:

        if line[0] == '[':
            if len(element_dict) != 0:     # appending the dict stored on previous iteration
                    final_list.append(element_dict)
                    element_dict = {} # again emtying dict
            element_dict['type'] = ''.join([i for i in line if i != '[' and i != ']'])
            
        else:
            val = line.split('=')
            element_dict[val[0].rstrip()] = val[1].lstrip()  #removing spaces on left and right side
        
    final_list.append(element_dict) # appending the values stored for last set
    return final_list


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
              
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def conv_parser(block,i,input_channels):
    # Create a sequential object
    layer = nn.Sequential()

    # Parse the properties for the convolution
    filters = int(block["filters"])
    kernel_size = int(block["size"])
    #padding = int(block["pad"])
    stride = int(block["stride"]) 

    padding = (kernel_size - 1) // 2

    # Parse the properties for the batch_normalization
    batch_norm = True if ("batch_normalize" in block) else False

    # Parse the properties for the activation function 
    activation_function = block["activation"]
    
    # Create & add the convolution in the block
    conv = nn.Conv2d(in_channels = input_channels, out_channels=filters, kernel_size=kernel_size, padding=padding, stride=stride, bias = not batch_norm)
    layer.add_module("conv_{0}".format(i),conv)

    # If batch_norm ==> add in the block as well
    if(batch_norm):
        layer.add_module("batch_norm_{0}".format(i),nn.BatchNorm2d(filters))

    # If activation is leaky ==> add activation function as well
    if(activation_function == "leaky"):
        layer.add_module("leaky_{0}".format(i),nn.LeakyReLU(0.1, inplace = True))
    return layer, filters

def upsample_parser(block,i):
    # Simple upsampling
    layer = nn.Sequential()
    layer.add_module("upsample_{0}".format(i),nn.Upsample(scale_factor=2, mode="bilinear"))
    return layer

def route_parser(block,i,output_filters):  

    # Creation of the sequential object
    layer = nn.Sequential()

    block['layers'] = block['layers'].split(',')
    #block['layers'][0] = int(block['layers'][0])

    start = int(block['layers'][0])
    if len(block['layers']) == 1:  
        filters = output_filters[i + start]
               

    elif len(block['layers']) == 2:
        indexEnd = int(block['layers'][1])
        block['layers'][1] = int(block['layers'][1]) - i 
        filters = output_filters[i + start] + output_filters[indexEnd]
          
    route = IdentityLayer()
    layer.add_module("route_{0}".format(i),route)
    return layer,filters

def shortcut_parser(block,i):

    # Create a sequential object
    layer = nn.Sequential()

    layer.add_module("shortcut_{0}".format(i),IdentityLayer())

    return layer

def yolo_parser(block,i):

    # Create a sequential object
    layer = nn.Sequential()

    mask = block["mask"].split(",")
    mask = [int(m) for m in mask]
    anchors = block["anchors"].split(",")
    anchors = [(int(anchors[j]), int(anchors[j + 1])) for j in range(0, len(anchors), 2)]
    anchors = [anchors[j] for j in mask]
    block["anchors"] = anchors
            
    detectorLayer = DetectionLayer(anchors)
    layer.add_module("Detection_{0}".format(i),detectorLayer)

    return layer

def create_base_model(blocks):

    # Parse the details of darknet
    darknet_details = blocks[0]

    # Number of input channels
    input_channels = 3
    # Neural network modules
    output_filters = []
    modulelist = nn.ModuleList()

    # Parse each block
    for i,block in enumerate(blocks[1:]):
        # Create a sequential object
        seq = nn.Sequential()

        if(block["type"] == "convolutional"):
            seq,output_filter = conv_parser(block,i,input_channels)
        elif(block["type"] == "upsample"):
            seq = upsample_parser(block,i)
        elif(block["type"] == "route"):
            seq,output_filter = route_parser(block,i,output_filters)
        elif(block["type"] == "shortcut"):
            seq = shortcut_parser(block,i)
        elif(block["type"] == "yolo"):
            seq = yolo_parser(block,i)

        modulelist.append(seq)
        output_filters.append(output_filter)
        input_channels = output_filter

    return darknet_details,modulelist


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_base_model(self.blocks)

    def forward(self, current_value, CUDA=False):

        # First block is only configuration
        modules = self.blocks[1:]

        outputs = {}  
        write = 0     

        # Iterate through all modules
        for i, module in enumerate(modules): 

            # Get the type of the module    
            module_type = (module["type"])

            # Simple forward block
            if module_type == "convolutional" or module_type == "upsample":
                current_value = self.module_list[i](current_value)
                outputs[i] = current_value

            # Shortcurt block
            elif  module_type == "shortcut":

                # Adding the two connection together and output the result
                current_value = torch.add(outputs[i-1],outputs[i+int(module["from"])])
                outputs[i] = current_value
            
            # Concatenation layer
            elif module_type == "route":
                
                # Get the index of the layers
                layers_index = [int(a) for a in module["layers"]]

                # Compute current value
                if len(layers_index) == 1:
                    current_value = outputs[i + layers_index[0]]
                if len(layers_index) == 2:
                    current_value = torch.cat((outputs[i + layers_index[0]],outputs[i + layers_index[1]]),1)
                outputs[i] = current_value
                
            # Output layer     
            elif module_type == 'yolo':

                # Get anchors 
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                input_dimension = int(self.net_info["height"])
                # Get the number of classes
                number_classes = int(module["classes"])

                # Apply the prediction transformation
                current_value = current_value.data   
                current_value = predictions(current_value,input_dimension,anchors,number_classes)
                
                # Write the results
                if not write:             
                    detections = current_value
                    write = 1
                else:       
                    detections = torch.cat((detections, current_value), 1)

                # Pass the values
                outputs[i] = outputs[i-1]


        # Return 0 if there are no present predictions 
        try:
            return detections   #return detections if present
        except:
            return 0

     
    def load_weights(self, weightfile):
        
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # Note: we dont have bias for conv when batch normalization is there