# nohup python npr_full_train.py > npr_full_train.log 2>&1 &
# ps -ef | grep npr_full_train.py # (kill pid from: jovyan    <pid>  220314  0 23:34 pts/7    00:00:00 sh ./transform_img2grad.sh 0 ../../datasets/ ./grads)




import wandb

wandb.login(key="79824aa6b958aeebce669281f175fe198eb060dd", relogin=True)

import os
import sys
import time
import torch
import torch.nn
import argparse
import wandb
from PIL import Image
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

import random
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


# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '../../CNNDetection_dataset/val/'
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.classes = []
    return val_opt


# params
opt = TrainOptions().parse()
opt.name = "4-class-resnet-100-epochs"
opt.dataroot = '../../CNNDetection_dataset/'
# opt.classes = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

opt.classes = ["car", "cat", "chair", "horse"]

opt.optim = "adam"
opt.batch_size = 32
opt.delr_freq = 10
opt.lr = 0.0002
opt.niter = 100
opt.save_epoch_freq = 20

seed_torch(100)
Testdataroot = os.path.join(opt.dataroot, 'test')
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
print('  '.join(list(sys.argv)) )
val_opt = get_val_opt()
Testopt = TestOptions().parse(print_options=False)
data_loader = create_dataloader(opt)


wandb.init(project='npr_full_train', name=opt.name + '_' +time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), config=opt)

model = Trainer(opt)

print(f'Model is running on device: {next(model.model.parameters()).device}')

def testmodel():
    print('*'*25);accs = [];aps = []
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    for v_id, val in enumerate(vals):
        Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
        Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
        Testopt.no_resize = False
        Testopt.no_crop = True
        acc, ap, _, _, _, _ = validate(model.model, Testopt)
        accs.append(acc);aps.append(ap)
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        wandb.log({f'{val}_acc': acc, f'{val}_ap': ap, 'step': model.total_steps})
    mean_acc = np.array(accs).mean()*100
    mean_ap = np.array(aps).mean()*100
    
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
    print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    wandb.log({'Mean_acc': mean_acc, 'Mean_ap': mean_ap, 'step': model.total_steps})


# model.eval();testmodel()
model.train()
print(f'cwd: {os.getcwd()}')
for epoch in range(opt.niter):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    wandb.log({'Epoch': epoch + 1, 'step': model.total_steps})


    for i, data in enumerate(data_loader):
        model.total_steps += 1
        epoch_iter += opt.batch_size

        model.set_input(data)
        model.optimize_parameters()

        if model.total_steps % opt.loss_freq == 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
            wandb.log({'Train_loss': model.loss, 'step': model.total_steps})


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
            (epoch, model.total_steps))
        model.save_networks('latest')
        model.save_networks(epoch)

    if epoch % opt.delr_freq == 0 and epoch != 0:
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                (epoch, model.total_steps))
        model.adjust_learning_rate()

    if epoch == 97:
        model.save_networks('98_epoch')

    # Validation
    model.eval()
    acc, ap = validate(model.model, val_opt)[:2]
    wandb.log({'Val_accuracy': acc, 'Val_ap': ap, 'step': model.total_steps, 'Epoch': epoch + 1})
    print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
    testmodel()
    model.train()

# model.eval();testmodel()
model.save_networks('last')
wandb.finish()