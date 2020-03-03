import json

with open("detect.json") as handle:
    config = handle.read()

config = json.loads(config)

enable_gpu = config['GPU'] == 1
print(enable_gpu)
classes = config['classes']

data_folder = config['data']