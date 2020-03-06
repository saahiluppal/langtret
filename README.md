# Create Your Custom Object Detection Model using architechure YOLOv3 for Keras.
<p>Your work is just to Download the dataset of the categories you want to detect and that's it.</p>
<p>This Project Uses <a href='https://github.com/pjreddie/darknet'>Darknet</a> for training and is highly inspired from <a href='https://github.com/experiencor/keras-yolo3'>Experiencor's project</a> For implementing Object Detection.</p>

## Downloading Dataset:
<p>First thing to implement an object detection is to download dataset. And downloading dataset is extremely easy.</p><p>Dataset will be downloaded from <a href='https://storage.googleapis.com/openimages/web/index.html'>Google's Open Image Dataset</a> which have wide variety of images available openly for sake of research and development</p>
<p>Downloading Each Image Manually will be Difficult but don't worry, We have solution for you. Just clone this <a href='https://github.com/theAIGuysCode/OIDv4_ToolKit'>repo</a> and you are good to go.</p>

```bash
$ git clone https://github.com/theAIGuysCode/OIDv4_ToolKit
```

Download Dataset

```bash
$ python main.py downloader --classes _class1_ _class2_ --type_csv train --limit [val] --multiclass 1
```
This command will download your dataset. Just replace _class1_ and _class2_ with classes of your choice. You can type as many classes you want. and replace --limit [val] with no of images you want to download. For example you want to download 100 images of each class, you will then type --limit 100.

For Example, we want to download the dataset of road vehicles, then our command will look like
 ```bash
 $ python main.py downloader --classes Car Van Truck Bus --type_csv train --limit 200 --multiclass 1
 ```
 Your dataset will be downloaded in cloned_repo/OID/Dataset/train/.
 
 Kindly Ensure that YOLO wants labels in particular format. So, edit classes.txt in cloned_repo and type each class in different line. Your classes.txt looks like this.
 ```bash 
 Car
 Van
 Truck
 Bus
 ```
