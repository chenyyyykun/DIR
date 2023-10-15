from __future__ import print_function
import os
import argparse
import numpy as np
import random
import itertools

import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from torchattacks import PGD #, AutoAttack, CW

# from model import ResNet18
from model import *
from lightweight_model import MAE_ViT
import time
# from models.wideresnet import WideResNet
# from models.preactresnet import PreActResNet18
# from models.vggnet import VGGNet19

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
parser.add_argument('--nat-label-train', type=str, help='natural training label', default='./data/train_labels.npy')
parser.add_argument('--nat-img-test', type=str, help='natural test data', default='./data/test_images.npy')
parser.add_argument('--nat-label-test', type=str, help='natural test label', default='./data/test_labels.npy')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=91, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='perturbation')
parser.add_argument('--num-steps', default=5,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/Standard_AT',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # optimizer.param_groups[0]['lr'] = 2e-5*((10*lr)**-1)
    optimizer.param_groups[1]['lr'] = lr

def train(args, model_mae, model, device, train_loader, optimizer, epoch):

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        batch_size = len(data)
        # calculate robust loss
        model.eval()
        model_mae.eval()
        data_adv = PGD(model_pre= model_mae,model=model, x_natural=data, y=label, 
                epsilon=args.epsilon, steps=args.num_steps, size=args.step_size)
       
        rec_data= model_mae(data_adv) # rec_data= model_mae(data)
        rec_adv = PGD_R(model=model, x_natural=rec_data,
                epsilon=args.epsilon, steps=args.num_steps, size=args.step_size)    
        
        model_mae.train()
        model.train()  
        
        optimizer.zero_grad()
                
        pre_data= model_mae(data_adv)
        logits_out = model(pre_data)
        loss_un = F.cross_entropy(logits_out, label)         

        rec_data= model_mae(data_adv) # rec_data= model_mae(data)
        logit_rec = model(rec_data)
        # criterion_kl = nn.KLDivLoss(reduction='sum')
        loss_rec = (1.0 / batch_size) * nn.KLDivLoss(reduction='sum')(F.log_softmax(model(rec_adv), dim=1),
                                F.softmax(logit_rec, dim=1))

        '''
        loss_rec = F.cross_entropy(model(rec_adv), label)         
        
        '''
                                                        
        loss = loss_un + 1 * loss_rec
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tloss_un:{:.6f}  loss_rec:{:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item(),loss_un.item(),loss_rec.item()))


def PGD(model_pre, model, x_natural, y, epsilon, steps, size):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for _ in range(steps):
            x.requires_grad_()
            with torch.enable_grad():
                pre_out = model_pre(x)
                logits = model(pre_out)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def PGD_R(model, x_natural, epsilon, steps, size):   
        # x_natural = x_natural.detach()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        for i in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = nn.KLDivLoss(size_average=False)(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + size * torch.sign(grad.detach())
            x = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x_adv, 0, 1)
        return x_adv

def eval_test(model_mae, model, device, test_loader):
    model.eval()
    model_mae.eval()
    correct = 0
    correct_adv = 0
    #with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)

        pre_out = model_mae(data)
        logits_out = model(pre_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data = PGD(model_pre=model_mae, model=model, x_natural=data, y=label, 
                epsilon=args.epsilon, steps=20, size=args.step_size)

        out = model_mae(data) 
        logits_out = model(out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

    print('Test: Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), correct_adv,
        len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))


def main():
    # settings
    setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # device_ids = [2,3]
    # setup data loader
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trans_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = datasets.CIFAR10(root='/data/cyk/dataset/', train=True, download=False, transform=trans_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = datasets.CIFAR10(root='/data/cyk/dataset/', train=False, download=False, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # init model, ResNet18() can be also used here for training
    model_mae = MAE_ViT(mask_ratio=0.5).to(device)
    model_mae = torch.load('/data/cyk/MAE-main/model/light_8_2.pt',map_location='cuda').module
    model_mae = torch.nn.DataParallel(model_mae)

    model = ResNet18().to(device)
    # model = PreActResNet18(10).to(device)
    # model = WideResNet(34, 10, 10).to(device)
    # model = VGGNet19().to(device)

    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #lr = 5e-6 10epoches:best_acc:36.47%,
    optimizer =  optim.SGD([{'params':model_mae.parameters(), 'lr':5e-5},{'params':model.parameters(),'lr':args.lr}], momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer =  optim.SGD(itertools.chain(model_mae.parameters(),model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        # adversarial training
        train(args, model_mae, model, device, train_loader, optimizer, epoch)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model_mae.state_dict(),'/data/cyk/MAE-main/model/ous_at_pgd5_1_mae.pth')
            torch.save(model.state_dict(),'/data/cyk/MAE-main/model/ous_at_pgd5_1.pth') #1.0精度48 1.1 精度 49
            print('save the model')
        torch.cuda.empty_cache()
        print('================================================================')

        # evaluation on natural examples
        print('================================================================')
    eval_test(model_mae, model, device, test_loader)
    print('using time:', time.time()-start_time)


if __name__ == '__main__':
    main()

