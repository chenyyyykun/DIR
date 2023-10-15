from __future__ import print_function
import os
import argparse
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import time

from model import ResNet18
from lightweight_model import MAE_ViT

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')

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
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/Mart',
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
    optimizer.param_groups[1]['lr'] = lr


def train(args, pre_model, model, device, train_loader, optimizer, epoch):
    kl = nn.KLDivLoss(reduction='none')
    beta = 6.0

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        # calculate robust loss
        model.eval()
        pre_model.eval()
        data_adv = PGD(pre_model, model=model, x_natural=data, y=label, 
                        epsilon=args.epsilon, steps=args.num_steps, size=args.step_size)

        model.train()
        pre_model.train()
        optimizer.zero_grad()

        batch_size = len(data)

        pre_data_nat = pre_model(data)
        logits = model(pre_data_nat)

        pre_data_adv = pre_model(data_adv)
        logits_adv = model(pre_data_adv)

        adv_probs = F.softmax(logits_adv, dim=1)

        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

        new_y = torch.where(tmp1[:, -1] == label, tmp1[:, -2], tmp1[:, -1])

        loss_adv = F.cross_entropy(logits_adv, label) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        nat_probs = F.softmax(logits, dim=1)

        true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + float(beta) * loss_robust

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))

def PGD(pre_model, model, x_natural, y, epsilon, steps, size):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(steps):
            x.requires_grad_()
            with torch.enable_grad():
                pre_out = pre_model(x)
                logits = model(pre_out)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x


def eval_test(pre_model, model, device, test_loader):
    pre_model.eval()
    model.eval()
    correct = 0
    correct_adv = 0
    #with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        pre_out = pre_model(data)
        logits_out = model(pre_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

        data = PGD(pre_model=pre_model,model=model, x_natural=data, y=label, 
                epsilon=args.epsilon, steps=20, size=args.step_size)
        pre_out= pre_model(data)
        logits_out = model(pre_out)
        pred = logits_out.max(1, keepdim=True)[1]
        correct_adv += pred.eq(label.view_as(pred)).sum().item()

    print('Test: Accuracy: {}/{} ({:.0f}%), Robust Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), correct_adv,
        len(test_loader.dataset), 100. * correct_adv / len(test_loader.dataset)))


def main():
    # settings
    setup_seed(args.seed)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:2" if use_cuda else "cpu")

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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    testset = datasets.CIFAR10(root='/data/cyk/dataset/', train=False, download=False, transform=trans_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    model_mae = MAE_ViT(mask_ratio=0.5).to(device)
    model_mae = torch.load('/data/cyk/MAE-main/model/light_1000_decoder1.pt')
    # model_mae = torch.nn.DataParallel(model_mae,device_ids=device_ids)

    model = ResNet18().to(device)

    # model = torch.nn.DataParallel(model,device_ids=device_ids)
    cudnn.benchmark = True
    optimizer = optim.SGD([{'params':model_mae.parameters(), 'lr':5e-5},{'params':model.parameters(),'lr':args.lr}],  momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        # adversarial training
        train(args, model_mae, model, device, train_loader, optimizer, epoch)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       '/data/cyk/MAE-main/model/mart_light.pth')
            print('save the model')

        print('================================================================')

    # evaluation on natural examples
        print('================================================================')
        eval_test(model_mae, model, device, test_loader)
        print('using time:', time.time()-start_time)

if __name__ == '__main__':
    main()
