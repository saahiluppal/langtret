import subprocess
import os
import json
import shutil

with open('detect.json') as handle:
    config = handle.read()

config = json.loads(config)

enable_gpu = config['GPU'] == 1
data_folder = config['data']
categories = config['classes']
num_categories = len(categories)
last_weights = config['weights']
backup_dir = config['backup']

print('#### Creating Model Script')
with open('modelrc.py') as handle:
    model_script = handle.read()
model_script = model_script.replace('yolo_filter = 255', 'yolo_filter = '+ str(int(num_categories + 5) * 3))
with open('make_model.py', 'w') as handle:
    handle.write(model_script)

clone_darknet = subprocess.Popen('git clone https://github.com/pjreddie/darknet'.split(), stdout=subprocess.PIPE)
output, error = clone_darknet.communicate()

def fn_gpu():
    with open("darknet/Makefile") as handle:
        makefile = handle.read()

    makefile = makefile.replace("GPU=0","GPU=1")
    makefile = makefile.replace("CUDNN=0","CUDNN=1")
    makefile = makefile.replace("OPENCV=0","OPENCV=1")

    print('GPU Enabled')

    with open("darknet/Makefile", 'w') as handle:
        handle.write(makefile)

if enable_gpu: fn_gpu()

print('#### Changing Directory to ./darknet')
os.chdir('darknet')

print('#### Makefile Running')
make = subprocess.Popen(['make'], stdout=subprocess.PIPE)
output, error = make.communicate()

print('#### Copying Data')
shutil.copytree(data_folder, 'data/obj/')

batches = max(4000, 2000 * num_categories)

print('#### Configuring YOLO cfg file')
with open('cfg/yolov3.cfg') as handle:
    yolo = handle.readlines()

for index in range(len(yolo)):
    if index == 2 or index == 3:
        yolo[index] = '#' + yolo[index]
    elif index == 5 or index == 6:
        yolo[index] = yolo[index].replace('#', '')
    elif index == 19:
        yolo[index] = 'max_batches = ' + str(batches) + '\n'
    elif index == 21:
        yolo[index] = 'steps=' + str(int(0.8 * (batches))) + ',' + str(int(0.9 * (batches))) + '\n'
    elif index == 782 or index == 695 or index == 609:
        yolo[index] = 'classes=' + str(num_categories) + '\n'
    elif index == 602 or index == 688 or index == 775:
        yolo[index] = 'filters=' + str((num_categories + 5) * 3) + '\n'

with open('cfg/yolov3_custom.cfg', 'w') as handle:
    handle.write(''.join(yolo))

print('#### Configuring Names file')
with open('data/obj.names', 'w') as handle:
    for cat in categories:
        handle.write(cat + '\n') 

print('#### Making backup directory')
try:
    os.mkdir(backup_dir)
except:
    print('backup directory already exists. Skipping...')

print('#### Making data file')
with open('data/obj.data', 'w+') as handle:
    handle.write('classes = ' + str(num_categories) + '\n')
    handle.write('train = data/train.txt\n')
    handle.write('valid = data/test.txt\n')
    handle.write('names = data/obj.names\n')
    handle.write('backup = backup/')

print('#### Generating train.txt file')
copy_file = subprocess.Popen('cp ../generate_train.py generate_train.py'.split(), stdout = subprocess.PIPE)
output, error = copy_file.communicate()

generate_train = subprocess.Popen('python generate_train.py'.split(), stdout = subprocess.PIPE)
output, error = generate_train.communicate()

if last_weights.endswith("darknet53.conv.74"):
    if not os.path.isfile(last_weights):
        print('#### Downloading pre-trained conv weights')
        conv_download = subprocess.Popen('wget https://pjreddie.com/media/files/darknet53.conv.74'.split(), stdout=subprocess.PIPE)
        output, error= conv_download.communicate()
else:
    last_weights = last_weights

train_string = './darknet detector train data/obj.data cfg/yolov3_custom.cfg ' + last_weights

print('#### Initializing Training')
train = subprocess.Popen(train_string.split(), stdout = subprocess.PIPE)
output, error = train.communicate()
