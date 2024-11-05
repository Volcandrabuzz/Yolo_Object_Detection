# Yolo_Object_Detection
# Object Detection with Yolov3


![cover](https://bitmovin.com/wp-content/uploads/2019/08/Object_detection_Blog_Image_Q3_19.jpg)

Object detection is a computer vision task that involves both localizing one or more objects within an image and classifying each object in the image.

It is a challenging computer vision task that requires both successful object localization in order to locate and draw a bounding box around each object in an image, and object classification to predict the correct class of object that was localized.
Yolo is a faster object detection algorithm in computer vision and first described by Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi in ['You Only Look Once: Unified, Real-Time Object Detection'](https://arxiv.org/abs/1506.02640)

This notebook implements an object detection based on a pre-trained model - [YOLOv3 Pre-trained Weights (yolov3.weights) (237 MB)](https://pjreddie.com/media/files/yolov3.weights).  The model architecture is called a “DarkNet” and was originally loosely based on the VGG-16 model.
