# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import csv
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils import progress_bar
from model.crate import *
from model.vit import *
from data.dataset import *


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default='adamW')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--data', default='cifar10')
parser.add_argument('--classes', type=int, default=10)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--randomaug', type=int, default=1)
parser.add_argument('--rand_aug_n', type=int, default=2)
parser.add_argument('--rand_aug_m', type=int, default=14)
parser.add_argument('--erase_prob', type=float, default=0.0)
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--patch', default='4', type=int, help='patch for ViT')
parser.add_argument('--ckpt_dir', type=str, default=None, help='location for the pretrained CRATE weight')
parser.add_argument('--data_dir', type=str, default='./data', help='location for datasets')
parser.add_argument('--gpu_ids', type=str, default=None, help='comma-separated gpu ids for DataParallel, e.g. 0,1,2,3')
parser.add_argument('--distributed', action='store_true', help='enable DDP; compatible with torchrun')
parser.add_argument('--dist_backend', type=str, default='nccl')
parser.add_argument('--workers', type=int, default=4, help='data loading workers per process')

parser.add_argument('--ole_mode', default='none', type=str, choices=['none', 'learned_t', 'solver_t'])
parser.add_argument('--ole_loss_weight', default=0.0, type=float)
parser.add_argument('--ole_lambda_sum', default=1.0, type=float)
parser.add_argument('--ole_layers', default='last3', type=str)
parser.add_argument('--ole_solver_step_size', default=0.1, type=float)
parser.add_argument('--ole_solver_second_order', action='store_true')
parser.add_argument('--ole_log_head_stats', action='store_true')
parser.add_argument('--nuclear_norm_mode', default='exact', type=str)

args = parser.parse_args()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_distributed():
    use_ddp = args.distributed or ('LOCAL_RANK' in os.environ)
    if use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend)
        return True, local_rank, torch.device(f'cuda:{local_rank}')
    if torch.cuda.is_available():
        return False, 0, torch.device('cuda')
    return False, 0, torch.device('cpu')


def build_model():
    if args.net == 'vit_tiny':
        model = vit_tiny_patch16(
            global_pool=True,
            ole_mode=args.ole_mode,
            ole_loss_weight=args.ole_loss_weight,
            ole_lambda_sum=args.ole_lambda_sum,
            ole_layers=args.ole_layers,
            ole_solver_step_size=args.ole_solver_step_size,
            ole_solver_second_order=args.ole_solver_second_order,
            ole_log_head_stats=args.ole_log_head_stats,
            nuclear_norm_mode=args.nuclear_norm_mode,
        )
        model.head = nn.Linear(192, args.classes)
    elif args.net == 'vit_small':
        model = vit_small_patch16(
            global_pool=True,
            ole_mode=args.ole_mode,
            ole_loss_weight=args.ole_loss_weight,
            ole_lambda_sum=args.ole_lambda_sum,
            ole_layers=args.ole_layers,
            ole_solver_step_size=args.ole_solver_step_size,
            ole_solver_second_order=args.ole_solver_second_order,
            ole_log_head_stats=args.ole_log_head_stats,
            nuclear_norm_mode=args.nuclear_norm_mode,
        )
        model.head = nn.Linear(384, args.classes)
    elif args.net == 'CRATE_tiny':
        model = CRATE_tiny(args.classes)
    elif args.net == 'CRATE_small':
        model = CRATE_small(args.classes)
    elif args.net == 'CRATE_base':
        model = CRATE_base(args.classes)
    elif args.net == 'CRATE_large':
        model = CRATE_large(args.classes)
    else:
        raise NotImplementedError(args.net)
    return model


use_amp = True
best_acc = 0
start_epoch = 0
is_distributed, local_rank, device = setup_distributed()

if is_main_process():
    print('==> Preparing data..')
    print(f'workers per process: {args.workers}')
size = 224
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((size, size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(args.rand_aug_n, args.rand_aug_m) if args.randomaug else transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=args.erase_prob),
])
transform_test = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transet, testset = load_dataset(
    args.data,
    size=size,
    transform_train=transform_train,
    transform_test=transform_test,
    data_dir=args.data_dir,
)

train_sampler = torch.utils.data.distributed.DistributedSampler(transet) if is_distributed else None
test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False) if is_distributed else None

batch_size = args.bs if not is_distributed else max(1, args.bs // dist.get_world_size())
loader_workers = max(args.workers, 0)
loader_kwargs = {
    'num_workers': loader_workers,
    'pin_memory': device.type == 'cuda',
    'persistent_workers': loader_workers > 0,
}
trainloader = torch.utils.data.DataLoader(
    transet,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    **loader_kwargs,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    sampler=test_sampler,
    **loader_kwargs,
)

if is_main_process():
    print('==> Building model..')
if args.ckpt_dir is None and is_main_process():
    print('Train from scratch.')

net = build_model()
if args.ckpt_dir is not None:
    state_dict = torch.load(args.ckpt_dir, map_location='cpu')['state_dict']
    for key in list(state_dict.keys()):
        if 'mlp_head' in key:
            del state_dict[key]
            if is_main_process():
                print('deleted:', key)
    net.load_state_dict(state_dict, strict=False)

if device.type == 'cuda':
    net = net.to(device)
    cudnn.benchmark = True

if is_distributed:
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
elif device.type == 'cuda':
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    if is_main_process():
        print(f'using data parallel on gpus: {gpu_ids}')
    if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

if args.resume:
    if is_main_process():
        print('==> Resuming from checkpoint..')
    checkpoint = torch.load(f'./checkpoint/{args.net}-ckpt.t7', map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    best_acc = checkpoint.get('acc', 0)
    start_epoch = checkpoint.get('epoch', 0)

criterion = nn.CrossEntropyLoss().to(device)
if args.opt == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
elif args.opt == 'adamW':
    if is_main_process():
        print('using adamW')
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
else:
    raise NotImplementedError(args.opt)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def get_model_ref(m):
    return m.module if hasattr(m, 'module') else m


def train(epoch):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    if is_main_process():
        print(f'\nEpoch: {epoch}')

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            task_loss = criterion(outputs, targets)
            ole_aux = outputs.new_zeros(())
            if args.net.startswith('vit') and args.ole_mode != 'none' and args.ole_loss_weight > 0:
                ole_aux = get_model_ref(net).get_aux_loss()
            loss = task_loss + args.ole_loss_weight * ole_aux

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if is_main_process():
            progress_bar(
                batch_idx,
                len(trainloader),
                'Task: %.3f | OLE: %.3f | Total: %.3f | Acc: %.3f%% (%d/%d)'
                % (task_loss.item(), ole_aux.item(), loss.item(), 100.0 * correct / max(total, 1), correct, total),
            )
            if args.net.startswith('vit') and args.ole_mode != 'none' and args.ole_log_head_stats and batch_idx % 100 == 0:
                head_stats = get_model_ref(net).get_ole_head_stats()
                if head_stats:
                    print(f'OLE head stats: {head_stats}')

    return train_loss / max(len(trainloader), 1)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if is_main_process():
                progress_bar(
                    batch_idx,
                    len(testloader),
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100.0 * correct / max(total, 1), int(correct), int(total)),
                )

    if is_distributed:
        stats = torch.tensor([test_loss, correct, total], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        test_loss, correct, total = stats.tolist()

    acc = 100.0 * correct / max(total, 1.0)
    if is_main_process() and acc > best_acc:
        print('Saving..')
        state = {
            'model': get_model_ref(net).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(state, f'./checkpoint/{args.net}-{args.patch}-ckpt.t7')
        best_acc = acc

    if is_main_process():
        os.makedirs('log', exist_ok=True)
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {acc:.5f}'
        print(content)
        with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
            appender.write(content + '\n')
    return test_loss, acc


list_loss = []
list_acc = []
for epoch in range(start_epoch, args.n_epochs):
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    scheduler.step(epoch - 1)
    if is_main_process():
        list_loss.append(val_loss)
        list_acc.append(acc)
        with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)

if is_distributed:
    dist.destroy_process_group()
