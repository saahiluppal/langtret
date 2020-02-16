import subprocess
import os

with open('objects.txt') as handle:
    objects = handle.readlines()

objects = list(map(lambda x: x.replace('\n', ''), objects))
num_objects = len(objects)

print(objects)
print(num_objects)

dataset_manager_string = 'python3 dataset_manager/main.py downloader --classes ' + ' '.join(objects) + '--type_csv train --limit ' + limit + ''

print(dataset_manager_string)