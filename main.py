import subprocess
import os
import shutil

enable_gpu = False
data_folder = '/home/anonymous/yolo_deps'
num_categories = 5
categories = ['Ambulance', 'Bus', 'Car', 'Truck', 'Van']

"""
clone_darknet = subprocess.Popen('git clone https://github.com/pjreddie/darknet'.split(), stdout=subprocess.PIPE)
output, error = clone_darknet.communicate()

def fn_gpu():
    with open("darknet/Makefile") as handle:
        makefile = handle.read()

    makefile = makefile.replace("GPU=0","GPU=1")
    makefile = makefile.replace("CUDNN=0","CUDNN=1")
    makefile = makefile.replace("OPENCV=0","OPENCV=1")

    with open("darknet/Makefile", 'w') as handle:
        handle.write(makefile)

if enable_gpu: fn_gpu()

os.chdir('darknet')

make = subprocess.Popen(['make'], stdout=subprocess.PIPE)
output, error = make.communicate()

shutil.copytree(data_folder, 'data/obj')

with open('cfg/yolov3.cfg') as handle:
    yolo = handle.readlines()

for index in range(len(yolo)):
    if index == 2 or index == 3:
        yolo[index] = '#' + yolo[index]
    elif index == 5 or index == 6:
        yolo[index] = yolo[index].replace('#', '')
    elif index == 19:
        yolo[index] = 'max_batches = ' + str(2000 * num_categories) + '\n'
    elif index == 21:
        yolo[index] = 'steps=' + str(int(0.8 * (2000 * num_categories))) + ',' + str(int(0.9 * (2000 * num_categories))) + '\n'
    elif index == 782 or index == 695 or index == 609:
        yolo[index] = 'classes=' + str(num_categories) + '\n'
    elif index == 602 or index == 688 or index == 775:
        yolo[index] = 'filters=' + str((num_categories + 5) * 3) + '\n'

with open('cfg/yolov3_custom.cfg', 'w') as handle:
    handle.write(''.join(yolo))

with open('data/obj.data', 'w') as handle:
    for cat in categories:
        handle.write(cat + '\n') 
"""