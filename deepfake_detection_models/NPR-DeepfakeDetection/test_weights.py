# nohup python test_weights.py > test_weights_100_epochs.log 2>&1 &
# ps -ef | grep test_weights.py



import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
import pandas as pd


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DetectionTests = {
                'ForenSynths': { 'dataroot'   : '../../datasets/ForenSynths/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

           'GANGen-Detection': { 'dataroot'   : '../../datasets/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

         'DiffusionForensics': { 'dataroot'   : '../../datasets/DiffusionForensics/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

        'UniversalFakeDetect': { 'dataroot'   : '../../datasets/UniversalFakeDetect/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

        'Diffusion1kStep': { 'dataroot'   : '../../datasets/Diffusion1kStep/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

                 }


weights_dir = "/home/jovyan/shares/SR006.nfs2/almas_deepfake_detection/deepfake_detection_models/NPR-DeepfakeDetection/checkpoints/4-class-resnet-100-epochs"
weights = os.listdir(weights_dir)

for w in weights:
    if w[-4:] == '.pth':
      opt = TestOptions().parse(print_options=False)

      # params
      opt.model_path = weights_dir + '/' + w
      opt.batch_size = 1024
      print(f'Model_path {opt.model_path}')
      torch.cuda.set_device('cuda:0')

      # get model
      model = resnet50(num_classes=1)
      model.load_state_dict(torch.load(opt.model_path, map_location='cuda:0'), strict=True)
      model.cuda()
      model.eval()

      results = []

      for testSet in DetectionTests.keys():
          dataroot = DetectionTests[testSet]['dataroot']
          printSet(testSet)

          accs = []
          aps = []
          print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
          for v_id, val in enumerate(os.listdir(dataroot)):
              opt.dataroot = '{}/{}'.format(dataroot, val)
              opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
              opt.no_resize = DetectionTests[testSet]['no_resize']
              opt.no_crop   = DetectionTests[testSet]['no_crop']
              acc, ap, _, _, _, _ = validate(model, opt)
              accs.append(acc)
              aps.append(ap)
              print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
              results.append({'group': testSet, 'id': v_id, 'dataset': val, 'accuracy': acc * 100, 'average_precision': ap * 100})

          results.append({'group': testSet, 'id': v_id + 1, 'dataset': 'Mean', 'accuracy': np.array(accs).mean() * 100, 'average_precision': np.array(aps).mean() * 100})    #TODO: изменить dataframe
          print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
          print('*'*25) 

      df = pd.DataFrame(results)
      df.to_csv(f'results/my_train_{w[:-4]}.csv', index=False)