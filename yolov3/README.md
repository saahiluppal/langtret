# Object Detection - YOLOv3
- This repo provides clean implementation of Object Detection Using Tensorflow and Keras under architecture YOLOv3 using all the best practices.
- This project is highly inspired by <a href='https://github.com/experiencor/keras-yolo3'>experiencor</a> and <a href='https://github.com/zzh8829/yolov3-tf2'>zzh8829</a> projects for implementation of YOLO.
<br />

## Requirements:
- tensorflow
- cv2
- moviepy


<br />

## Model Creation:
Grab the pretrained weights of yolo3 from
<a href='https://pjreddie.com/media/files/yolov3.weights'>here</a>.<br /><br />
<strong>Or Hit:</strong>
```bash
$ wget https://pjreddie.com/media/files/yolov3.weights
```
<strong>Then Run the script:</strong>
```bash
$ python3 make_model.py
```
<p>After completion of the script, a model will be created named <strong>model.h5</strong></p>
<br />

## Detection:
For simple Object detection on an image:
```bash
$ python3 detect_image.py -i /path/to/image
```
For Object detection on a pre-saved video:
```bash
$ python3 detect_video.py -f /path/to/video_file [-o  /path/to/output/video_file] [-s subclip [from] [to]]
```
For Object detection on live camera stream:
```bash
$ python3 live.py
```
<br />

- Object Detection on pre-saved video may be slow. But the script will save the video with objects detected so that one can see video without delays.
- Object Detection on live camera feed might be slow.
<br />

## License:
<a href='https://github.com/saahiluppal/object_detection/blob/master/yolov3/LICENSE'>MIT License</a>
