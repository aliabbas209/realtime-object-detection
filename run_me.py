from Detector import *

#SSD MobileNet v2 320x320 Speed (ms)=19 	COCO mAP=20.2
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

#EfficientDet D0 512x512 Speed (ms)=39 	COCO mAP=33.6
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz"

#EfficientDet D1 640x640 Speed (ms)=54 	COCO mAP=38.4
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz"


#EfficientDet D4 1024x1024 Speed (ms)=133 	COCO mAP=48.5
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

#Faster R-CNN ResNet50 V1 640x640 Speed (ms)=53 	COCO mAP=29.3
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
imagePath = "test/1.jpg"
videoPath = 0
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)

detector.downloadModel(modelURL)
detector.loadModel()
#detector.predictImage(imagePath, threshold)

detector.predictVideo(videoPath, threshold)

