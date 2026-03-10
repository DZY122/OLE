import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch

from model.vit import vit_tiny_patch16


def run_mode(mode):
    model = vit_tiny_patch16(
        global_pool=True,
        num_classes=10,
        ole_mode=mode,
        ole_loss_weight=1.0,
        ole_layers='last3',
        ole_lambda_sum=1.0,
        ole_solver_step_size=0.1,
        ole_solver_second_order=False,
    )
    model.train()
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10), f'bad logits shape for {mode}: {logits.shape}'
    aux = model.get_aux_loss()
    if mode == 'none':
        assert aux.item() == 0.0
    else:
        assert torch.isfinite(aux), f'aux not finite for {mode}'
    loss = logits.mean() + (aux if mode != 'none' else 0.0)
    loss.backward()


if __name__ == '__main__':
    for m in ['none', 'learned_t', 'solver_t']:
        run_mode(m)
    print('OLE smoke test passed')
