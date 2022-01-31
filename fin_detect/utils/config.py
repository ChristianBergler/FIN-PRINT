"""
Module: config.py
Authors: Christian Bergler, Alexander Gebhard
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 25.01.2022
"""

"""
Code from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/pytorchyolo/utils/parse_config.py
License: GNU General Public License v3.0
Access Data: 06.06.2020, Last Access Date: 25.01.2022
Changes: Modified by Christian Bergler, Alexander Gebhard (continuous since 06.06.2020)
"""

"""
Reads, converts, and parses YOLOv3 config file using the DarkNet53 backbone
"""
def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

"""
Reads, converts, and parses YOLOv3 data config file including data split, class name, number of class information
"""
def parse_data_config(path):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

"""
Updates an existing info.data config based on the given options dictionary
"""
def write_to_data_config(path, options):
    try:
        with open(path, 'w') as fp:
            for key in options:
                if key == "gpus" or key == "num_workers":
                    continue
                else:
                    value = options.get(key)
                fp.write(key+"="+value+"\n")
        return True
    except Exception:
        return False