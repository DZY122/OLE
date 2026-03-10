import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.crate import CRATE_base, CRATE_large, CRATE_small, CRATE_tiny
from model.vit import vit_small_patch16, vit_tiny_patch16


def build_model(args):
    if args.arch == 'vit_tiny':
        model = vit_tiny_patch16(global_pool=True, num_classes=args.num_classes)
    elif args.arch == 'vit_small':
        model = vit_small_patch16(global_pool=True, num_classes=args.num_classes)
    elif args.arch == 'CRATE_tiny':
        model = CRATE_tiny(args.num_classes)
    elif args.arch == 'CRATE_small':
        model = CRATE_small(args.num_classes)
    elif args.arch == 'CRATE_base':
        model = CRATE_base(args.num_classes)
    elif args.arch == 'CRATE_large':
        model = CRATE_large(args.num_classes)
    else:
        raise NotImplementedError(args.arch)
    return model


def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state, strict=False)


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    parser = argparse.ArgumentParser('Multi-GPU inference/eval')
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arch', default='vit_tiny', choices=['vit_tiny', 'vit_small', 'CRATE_tiny', 'CRATE_small', 'CRATE_base', 'CRATE_large'])
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist-backend', default='nccl')
    args = parser.parse_args()

    distributed = args.distributed and 'LOCAL_RANK' in os.environ
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args)
    load_checkpoint(model, args.checkpoint, device)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    elif device.type == 'cuda':
        if args.gpu_ids is not None:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        model = model.to(torch.device(f'cuda:{gpu_ids[0]}'))
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        model = model.to(device)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
    )

    model.eval()
    top1_total = 0.0
    top5_total = 0.0
    n_total = 0
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            bs = images.size(0)
            top1_total += acc1.item() * bs
            top5_total += acc5.item() * bs
            n_total += bs

    if distributed:
        total = torch.tensor([top1_total, top5_total, n_total], device=device, dtype=torch.float64)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        top1_total, top5_total, n_total = total.tolist()

    if (not distributed) or dist.get_rank() == 0:
        print(f'Top1: {top1_total / n_total:.3f}  Top5: {top5_total / n_total:.3f}  Samples: {int(n_total)}')

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
