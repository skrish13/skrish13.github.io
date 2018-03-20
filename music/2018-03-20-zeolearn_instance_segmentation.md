---
layout: post
excerpt: ""
categories: [music]
comments: False
permalink: /music/zeo/
---

## Instance Segmentation using Deep Learning

As we all know, object detection is the task of detecting objects in an image in the form of a bounding box. What if we wanted to get a more accurate information about the object? You'd go for more than a rectangle (bounding box), maybe a polygon which represents the object more tightly. But that's still not the best way. The best way would be to assign each pixel inside the bounding box which actually has the object. This task is called as Instance segmentation, where you segment the object instances.

In this post, we are going to look in depth at a state of the art (SOTA) method which does Instance Segmentation using deep learning. It's called **Mask R-CNN** [3], published by the Facebook AI Research (FAIR) team at ICCV 2017. The post assumes a basic understanding of deep learning and CNNs for object detection. For easier understanding, I'll be using examples of code samples in PyTorch as its pretty popular these days. The excellent Keras implementation is also given in the references [6]

### Core Idea

It builds on the very popular method for object detection, Faster R-CNN. They add another head (branch) for the segmentation task. This makes the total branches to be 3 - classification, bounding box regression and segmentation. They also enhance the ROIPooling step in FasterRCNN and propose a ROIAlign layer instead. We won't go into details of Faster R-CNN in this post but enough details will be explained for understanding of Mask-RCNN.

### Objective

The focus of the authors is on using simple and basic network design to show the efficiency of the idea/concept. They get the SOTA without any complimentary techniques (eg: OHEM, multi-scale train/test etc). These can be used to further improve accuracy very easily. This isn't in the scope of the paper.

### Backbones - ResNets, FPNs and Faster R-CNN

- It's a two-stage network, just like Faster R-CNN. The first stage is region proposal network (RPN) and the second stage is the combined object detection, segmentation network. 
- The first-stage is exactly identical to Faster R-CNN. The RPN is proposed and explained in depth in the Faster R-CNN paper [2].
- The second stage has two parts 
  - Feature Extractor
  - Task Specific Heads (branches)
- The feature extractor as the name suggests is interchangeable and serves as a backbone to extract features. A very popular feature extractor used to be VGG [5] network which was used in the Faster R-CNN paper few years ago. But better feature extractors have come up recently, namely ResNets and more recently Feature Pyramid Networks (FPNs) which builds on older ResNets. The details of the networks is beyond the scope of this post.
- The task specific heads are parallel networks which are trained together. A code sample is shown below. It is taken from the Faster R-CNN code in PyTorch [3]
```python
self.fc6 = FC(512 * 7 * 7, 4096)
self.fc7 = FC(4096, 4096)
self.score_fc = FC(4096, self.n_classes, relu=False)
self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)
```

- Here, `fc6` and `fc7` are simple Fully Connected Layers, while `score_fc` and `bbox_fc` are corresponding predictors for classification score and bounding box co-ordinates (or offsets). These are referred to as heads or branches. (Note that both the predictors operate on the same features, which comes from `fc7`)
- Here, Loss is sum of classification loss ($L_{cls}$) and bounding box loss ($L_{box}$), where $L_{cls}$ is $CrossEntropyLoss$ and $L_{box}$ is $SmoothL1Loss$. 

### Mask Head

- One of the main contribution of the paper is the addition of the Mask head to do the instance segmentation task. This is fully convolutional network unlike the other heads which are FC layers.

- The output of the segmentation task should be a segmentation map big enough to represent an object of average size. The network architecture is taken from the paper and is shown below.

  ![Mask Head](img/mask head.png)

- Let's take the FPN backbone for explanation (similar logic applies for ResNet as well)

- The output feature maps from ResNet is passed as input to a stack of four convolution layers with constant number of feature maps (256) with a deconvolution layer (size=2) in the end to increase the spatial resolution from 14x14 to 28x28. The last (output) conv is a 1x1 convolution with number of feature maps as number of classes.

- A sample code to better understand above. This is a PyTorch Mask R-CNN code taken from [4]. Batch normalization is a normalization layer which is used after most conv layers to help in training faster, being more stable etc.

```python
# Setup layers
self.mrcnn_mask_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.mrcnn_mask_bn1 = nn.BatchNorm2d(256, eps=0.001)

self.mrcnn_mask_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.mrcnn_mask_bn2 = nn.BatchNorm2d(256, eps=0.001)

self.mrcnn_mask_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.mrcnn_mask_bn3 = nn.BatchNorm2d(256, eps=0.001)

self.mrcnn_mask_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
self.mrcnn_mask_bn4 = nn.BatchNorm2d(256, eps=0.001)

self.mrcnn_mask_deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

self.mrcnn_mask = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)
```

- The Mask loss ($L_{mask}$) is again $CrossEntropy$. So the total loss is the sum of $L_{cls}, L_{box}, L_{mask}$ 
- The network is trained simultaneously on all three heads.

### ROI Align

- One of their other important contribution is the ROIAlign Layer instead of ROIPool (in Faster R-CNN). This basically doesn't round off your (x/spatial_scale) fraction to an integer (like it does in the case of ROIPool). Instead it does bilinear interpolation to find out the pixels at those floating values.
- The same process is used to get floating point value instead of integers (quantization) while assigning spatial portions into output bins in ROIPooling
- For example: Let's assume ROI height and width is 54,167 respectively. 
- **Spatial scale** is basically Image size/FeatureMap size (H/h, W/w), its also called **stride** in this context. Usually its a square, so we just use one notation. 
- Let's assume its H=224, h=14. This gives the spatial scale as 16.
- Dimensions of the corresponding portion in the output feature map
  - ROIPool: 54/16, 167/16 = **3,10**
  - ROIAlign: 54/16, 167/16 = **3.375, 10.4375**
  - Now we can use bilinear interpolation to get upsample it and get exact pixel values of those positions and not lose the 0.375\*16 and 0.4375\*16
- Similar logic goes into seperating the corresponding the regions into appropriate bins according to the ROIAlign output shape (eg 7x7).
- Code example is given below from [5]

```python
def ROIAlign(feature_maps, rois, config, pool_size, mode='bilinear'):
    """Implements ROI Align on the features.
    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels
    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]
    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    #feature_maps= [P2, P3, P4, P5]
    rois = rois.detach()
    crop_resize = CropAndResize(pool_size, pool_size, 0)
    
    roi_number = rois.size()[1]
    
    pooled = rois.data.new(
            config.IMAGES_PER_GPU*rois.size(
            1), 256, pool_size, pool_size).zero_()
            
    rois = rois.view(
            config.IMAGES_PER_GPU*rois.size(1),
            4)
                   
    # Loop through levels and apply ROI pooling to each. P2 to P5.
    x_1 = rois[:, 0]
    y_1 = rois[:, 1]
    x_2 = rois[:, 2]
    y_2 = rois[:, 3]


    roi_level = log2_graph(
        torch.div(torch.sqrt((y_2 - y_1) * (x_2 - x_1)), 224.0))
        
        
    roi_level = torch.clamp(torch.clamp(
        torch.add(torch.round(roi_level), 4), min=2), max=5)

    # P2 is 256x256, P3 is 128x128, P4 is 64x64, P5 is 32x32
    # P2 is 4, P3 is 8, P4 is 16, P5 is 32
    for i, level in enumerate(range(2, 6)):

        scaling_ratio = 2**level

        height = float(config.IMAGE_MAX_DIM)/ scaling_ratio
        width = float(config.IMAGE_MAX_DIM) / scaling_ratio

        ixx = torch.eq(roi_level, level)

        box_indices = ixx.view(-1).int() * 0
        ix = torch.unsqueeze(ixx, 1)
        level_boxes = torch.masked_select(rois, ix)
        if level_boxes.size()[0] == 0:
            continue
        level_boxes = level_boxes.view(-1, 4)
        
        crops = crop_resize(feature_maps[i], torch.div(
                level_boxes, float(config.IMAGE_MAX_DIM)
                )[:, [1, 0, 3, 2]], box_indices)
                
        indices_pooled = ixx.nonzero()[:, 0]
        pooled[indices_pooled.data, :, :, :] = crops.data

    pooled = pooled.view(config.IMAGES_PER_GPU, roi_number,
               256, pool_size, pool_size)        
    pooled = Variable(pooled).cuda()
    return pooled
```



### Other Experiments 

Lots of explanation and ablation studies proving the statements is given in the paper. 

- Usage of multinomial masks vs individual masks (softmax vs sigmoid). The output of the Mask Head can be a K-way classifying softmax output or K-way independent sigmoidal output. It's shown that independent outputs outperform softmax.
- Using the information from box head and just predicting the extent of the object instead of classifying each pixel as described above makes the model easier to train. In this case it'd be just be a binary mask (object or not) as the class information is taken from other branches.
- Using FCNs (fully convolutional network) for segmentation task gives a decent boost in accuracy as expected. Conv layers perform much better in predicting image masks than fully connected layers.
- Using ROIAlign in place of ROIPool helps increasing the accuracy by a huge margin

## References

```
[1] He, Kaiming, Georgia Gkioxari, Piotr Dollár and Ross B. Girshick. “Mask R-CNN.” *2017 IEEE International Conference on Computer Vision (ICCV)* (2017): 2980-2988.

[2] Ren, Shaoqing, Kaiming He, Ross B. Girshick and Jian Sun. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” *IEEE Transactions on Pattern Analysis and Machine Intelligence* 39 (2015): 1137-1149.

[3] "Faster R-CNN, PyTorch", https://github.com/longcw/faster_rcnn_pytorch

[4] "Mask R-CNN, PyTorch", https://github.com/soeaver/Pytorch_Mask_RCNN

[5] Simonyan, Karen and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” CoRR abs/1409.1556 (2014): n. pag.

[6] "Mask R-CNN, Keras", https://github.com/matterport/Mask_RCNN
```
