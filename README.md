# Yolo_Object_Detection
# Object Detection with Yolov3


![cover](https://bitmovin.com/wp-content/uploads/2019/08/Object_detection_Blog_Image_Q3_19.jpg)

Object detection is a computer vision task that involves both localizing one or more objects within an image and classifying each object in the image.

It is a challenging computer vision task that requires both successful object localization in order to locate and draw a bounding box around each object in an image, and object classification to predict the correct class of object that was localized.
Yolo is a faster object detection algorithm in computer vision and first described by Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi in ['You Only Look Once: Unified, Real-Time Object Detection'](https://arxiv.org/abs/1506.02640)

This notebook implements an object detection based on a pre-trained model - [YOLOv3 Pre-trained Weights (yolov3.weights) (237 MB)](https://pjreddie.com/media/files/yolov3.weights).  The model architecture is called a “DarkNet” and was originally loosely based on the VGG-16 model.

## Inference on images


![yolo_img_infer_1](https://user-images.githubusercontent.com/26242097/48849319-15d21180-edcc-11e8-892f-7d894be8d1a6.png)
![yolo_img_infer_2](https://user-images.githubusercontent.com/26242097/48850723-41a2c680-edcf-11e8-8940-aec302cd8aa3.png)
![yolo_infer_3](https://user-images.githubusercontent.com/26242097/48850729-449db700-edcf-11e8-853d-9f3eca6233c9.png)
![yolo_img_infer_4](https://user-images.githubusercontent.com/26242097/48850735-47001100-edcf-11e8-80d6-b474e54fd7af.png)
