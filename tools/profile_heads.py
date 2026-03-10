import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import json

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.vit import vit_tiny_patch16, vit_small_patch16
from model.ole_utils import effective_rank


def principal_angle_similarity(xi: torch.Tensor, xj: torch.Tensor, k: int) -> float:
    ui = torch.linalg.svd(xi, full_matrices=False).U[:, :k]
    uj = torch.linalg.svd(xj, full_matrices=False).U[:, :k]
    s = torch.linalg.svdvals(ui.transpose(0, 1) @ uj)
    return s.mean().item()


def parse_layers(ole_layers: str, depth: int):
    if ole_layers == 'last3':
        return list(range(max(depth - 3, 0), depth))
    if ole_layers == 'all':
        return list(range(depth))
    return [int(x.strip()) for x in ole_layers.split(',') if x.strip()]


def build_model(arch, num_classes, args):
    common = dict(
        global_pool=True,
        num_classes=num_classes,
        ole_mode=args.ole_mode,
        ole_lambda_sum=args.ole_lambda_sum,
        ole_layers=args.ole_layers,
        ole_solver_step_size=args.ole_solver_step_size,
        nuclear_norm_mode=args.nuclear_norm_mode,
    )
    if arch == 'vit_tiny':
        return vit_tiny_patch16(**common)
    if arch == 'vit_small':
        return vit_small_patch16(**common)
    raise ValueError('Only vit_tiny/vit_small supported for profiling utility')


def main():
    parser = argparse.ArgumentParser('Profile attention head redundancy')
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arch', default='vit_tiny', choices=['vit_tiny', 'vit_small'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=10)
    parser.add_argument('--ole_mode', default='none', choices=['none', 'learned_t', 'solver_t'])
    parser.add_argument('--ole_layers', default='last3')
    parser.add_argument('--ole_lambda_sum', type=float, default=1.0)
    parser.add_argument('--ole_solver_step_size', type=float, default=0.1)
    parser.add_argument('--nuclear_norm_mode', default='exact')
    parser.add_argument('--principal-k', type=int, default=8)
    parser.add_argument('--out-json', default='head_profile.json')
    parser.add_argument('--out-npz', default='head_profile.npz')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    valdir = os.path.join(args.data, 'val')
    val_dataset = datasets.ImageFolder(valdir, transform)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.arch, len(val_dataset.classes), args).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    depth = len(model.blocks)
    layers = parse_layers(args.ole_layers, depth)

    per_layer_heads = {str(l): [] for l in layers}
    with torch.no_grad():
        for bi, (images, _) in enumerate(loader):
            if bi >= args.max_batches:
                break
            images = images.to(device)
            _ = model(images)
            for l in layers:
                heads = model.blocks[l].attn.last_out_heads
                if heads is None:
                    continue
                hfeat = heads.permute(1, 0, 2, 3).reshape(heads.shape[1], -1, heads.shape[-1]).cpu()
                per_layer_heads[str(l)].append(hfeat)

    results = {}
    npz_payload = {}
    for lk, chunks in per_layer_heads.items():
        if not chunks:
            continue
        feats = torch.cat(chunks, dim=1)
        h = feats.shape[0]
        d = feats.shape[-1]
        k = min(args.principal_k, d)

        flat = feats.reshape(h, -1)
        flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
        cos_mat = flat @ flat.transpose(0, 1)
        mean_pair = ((cos_mat.sum() - torch.diag(cos_mat).sum()) / max(h * (h - 1), 1)).item()

        pa = torch.zeros(h, h)
        for i in range(h):
            for j in range(h):
                pa[i, j] = principal_angle_similarity(feats[i], feats[j], k)

        eranks = [effective_rank(feats[i]) for i in range(h)]
        sum_erank = effective_rank(feats.sum(dim=0))

        results[lk] = {
            'mean_pairwise_cosine': mean_pair,
            'effective_rank_per_head': eranks,
            'effective_rank_sum_heads': sum_erank,
            'principal_angle_similarity': pa.tolist(),
            'cosine_similarity_matrix': cos_mat.tolist(),
        }
        npz_payload[f'layer_{lk}_pa'] = pa.numpy()
        npz_payload[f'layer_{lk}_cos'] = cos_mat.numpy()

    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(args.out_npz, **npz_payload)
    print(f'saved {args.out_json} and {args.out_npz}')


if __name__ == '__main__':
    main()
