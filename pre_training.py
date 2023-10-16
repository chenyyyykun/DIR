import numpy as np
from PIL import Image
import argparse
from utils import setup_seed
from os import listdir
import math
from tqdm import tqdm
import sys
import pickle
import random
import torch
import cv2
from model import ResNet18
import torch.nn as nn
import torchvision.transforms as transforms

from model_try import *
# from model import *

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--total_epoch', type=int, default=1000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    # parser.add_argument('--model_path', type=str, default='/data/DIR-main/model/average_1.4_1000.pt')  
    parser.add_argument('--num_steps',type=int,default=20)              
    parser.add_argument('--epsilon', type=float, default=0.0314,help='perturbation')
    parser.add_argument('--step_size', type=float, default=0.003,help='perturb step size')

    return parser


def unpickle(file):
    read = open(r"/data/dataset/cifar-ours/"+file,'rb')
    dict = pickle.load(read)
    return dict

data_set=["nature_sample.pkl","adversarial_sample.pkl","nature_sample_test.pkl","adversarial_sample_test.pkl"]
nat_sample = unpickle(data_set[0])
adv_sample = unpickle(data_set[1])
nat_sample_test = unpickle(data_set[2])
adv_sample_test = unpickle(data_set[3])


def main(args):
    setup_seed(args.seed)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # device_ids = [2,3]
    model = DIR_ViT(mask_ratio=args.mask_ratio)
    model = torch.load('/data/DIR-main/model/average_1.4.pt')
    # model = torch.nn.DataParallel(model,device_ids=device_ids)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        losses_test = []
        nat_data=nat_sample[b'data']
        adv_data=adv_sample[b'data']
        nat_data = nat_data.reshape(50000,3,32,32)
        adv_data = adv_data.reshape(50000,3,32,32)
        batches = tqdm(range(int(np.shape(nat_data)[0]/args.batch_size)))
        for b in batches:
            data_input = torch.tensor(np.float32(((adv_data[b*args.batch_size:(b+1)*args.batch_size,:]/255)-0.5)*2))
            data_output = torch.tensor(np.float32(((nat_data[b*args.batch_size:(b+1)*args.batch_size,:]/255)-0.5)*2))
            data_input, data_output = data_input.to(device), data_output.to(device)
            _, loss = model(data_input,data_output)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            # print('batch[{}/{}]'.format(b+1,int(np.shape(nat_data)[0]/args.batch_size)))
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        # writer.add_scalar('DIR_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            nat_data_test=nat_sample_test[b'data']
            adv_data_test=adv_sample_test[b'data']
            nat_data_test = nat_data_test.reshape(10000,3,32,32)
            adv_data_test = adv_data_test.reshape(10000,3,32,32)
            for b in range(int(np.shape(nat_data_test)[0]/args.batch_size)):
                data_input = torch.tensor(np.float32(((adv_data_test[b*args.batch_size:(b+1)*args.batch_size,:]/255)-0.5)*2))
                data_output = torch.tensor(np.float32(((nat_data_test[b*args.batch_size:(b+1)*args.batch_size,:]/255)-0.5)*2))
                data_input, data_output = data_input.to(device), data_output.to(device)
                _, loss_test = model(data_input, data_output)
            #     loss_test = torch.mean((predicted_img - data_output) ** 2 )
                losses_test.append(loss_test.item())
            avg_loss_test = sum(losses_test) / len(losses_test)
            print(f'In epoch {e}, average traning loss in test is {avg_loss_test}.')
        
        ''' save model '''
        torch.save(model, args.model_path)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
