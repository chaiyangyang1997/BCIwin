import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from visdom import Visdom
from model.LightConvNet import LightConvNet
from model.baseModel import baseModel
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset
import json

def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def write_list_to_file(file_path, my_list):
#     with open(file_path, 'w') as file:
#         for item in my_list:
#             file.write(str(item) + '\n')

def main(config):
    dataPath = config['dataPath']
    outFolder = config['outFolder']
    randomFloder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    
    lr = config['lr']
    lrFactor = config['lrFactor']
    lrPatience = config['lrPatience']
    lrMin = config['lrMin']

    for subId in range(1, 10):
        trainDataFiles = ['A0' + str(subId) + 'T']
        validationSet = config['validationSet']
        testDataFile = ['A0' + str(subId) + 'E']

        outPath = os.path.join(outFolder, config['network'], 'sub'+str(subId), randomFloder)
        
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        
        print("Results will be saved in folder: " + outPath)

        dictToYaml(os.path.join(outPath, 'config.yaml'), config)

        setRandom(config['randomSeed'])

        data, labels = load_BCI42a_data(dataPath, trainDataFiles)
        trainData, trainLabels, valData, valLabels = split_data(data, labels, validationSet)
        testData, testLabels = load_BCI42a_data(dataPath, testDataFile)

        trainDataset = eegDataset(trainData, trainLabels)
        valDataset = eegDataset(valData, valLabels)
        testDataset = eegDataset(testData, testLabels)

        # vis = Visdom()

        netArgs = config['networkArgs']
        net = eval(config['network'])(**netArgs)
        print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

        fmap_block = list()
        input_block = list()

        def forward_hook(module, data_input, data_output):
            fmap_block.append(data_output)
            input_block.append(data_input)

        net.conv.register_forward_hook(forward_hook)

        lossFunc = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=lrFactor, patience=lrPatience, min_lr=lrMin)

        all_random_labels = list()

        model = baseModel(net, config, all_random_labels, resultSavePath=outPath)

        model.train(trainDataset, valDataset, lossFunc, optimizer, all_random_labels, scheduler=scheduler)

        classes = ['left hand', 'right hand', 'foot', 'tongue']
        model.test(testDataset, classes, all_random_labels)

        data_random_outFolder = "data/dataset/bci_iv_2a/random_data"

        random_data_save_path = os.path.join(data_random_outFolder, config['network'], 'sub'+str(subId), randomFloder)
        if not os.path.exists(random_data_save_path):
            os.makedirs(random_data_save_path)

        fmap_block_numpy = list()

        for i in range(len(fmap_block)):

            fmap_block_numpy.append(fmap_block[i].view(fmap_block[i].size(0), -1).cpu().detach().numpy())

        data_random_files = 'A0' + str(subId) + '_random_data.npy'

        label_random_files = 'A0' + str(subId) + '_random_label.npy'

        random_save_path_data = os.path.join(random_data_save_path, data_random_files)

        random_save_path_label = os.path.join(random_data_save_path, label_random_files)


        np.save(random_save_path_data, fmap_block_numpy)

        np.save(random_save_path_label, all_random_labels)

        # print("Person" + str(subId) + "feature maps shape: {}\n".format(len(fmap_block)))
        # for i in range(0, len(fmap_block)):
        #     print("Person" + str(subId) + "feature maps shape: {}\n".format(fmap_block[i].shape))
        #
        # print("Person" + str(subId) + "all_random_labels shape: {}\n".format(len(all_random_labels)))


        # vis.close()

if __name__ == '__main__':
    configFile = 'config/bciiv2a_config.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)

