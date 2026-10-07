"""
Staged finetuning on top of an existing finetuned EEGMamba baseline.

  stage 1 : train only the new gated branch (default, --mode gated)
            or the whole patch encoder (--mode full_pe)
  stage 2 : unfreeze everything at low LR

Arms for the comparison:
  C  (proposed) : --mode gated
  C' (switch)   : --mode full_pe
  B  (control)  : --mode full_pe --no_new_branch
  A  (continue) : --no_new_branch --stage1_epochs 0
Check equivalence to your baseline first with --verify.

"""
import argparse
import copy
import json
import os
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# This file lives in the EEGMamba repo root (next to models/ and modules/); run from there:
#   python finetune_staged.py ...
# model_ms.py lives inside models/ (next to eegmamba.py and model_for_isruc.py).
from models.model_for_isruc import Model as BaselineModel
from models.model_ms import ModelMS, load_baseline, make_optimizer, set_stage


# ----------------------------------------------------------------------------
# DATA: the single integration point. Return three DataLoaders that yield
# (x, y) with x: [B, 20, 6, 6000] float, y: [B, 20] long (same as your trainer).
# ----------------------------------------------------------------------------
AGE_BINS = ['1-2y', '3-5y', '6-12y', '13-18y', '19-100y']   # must match the manifest / list file names


def parse_age_bins(spec):
    """'all' | '6-12y' | '3-5y,6-12y' -> explicit list of bins (aggregated into one train/val/test).

    'all' is expanded to the five explicit bins (not "whatever list files exist"), so the cache is
    only used if EVERY bin has lists; otherwise the whole run uses the live path (never a partial cache).
    """
    spec = str(spec).strip()
    bins = list(AGE_BINS) if spec.lower() == 'all' else [b.strip() for b in spec.split(',') if b.strip()]
    bad = [b for b in bins if b not in AGE_BINS]
    if bad:
        raise ValueError(f'Unknown age bin(s) {bad}; valid: {AGE_BINS}, a comma list, "all", or "each".')
    return list(dict.fromkeys(bins))


def get_loaders(args):
    from datasets.nch_cached_dataset import LoadDatasetCachedFirst  # same package as nch_dataset.py
    params = Namespace(datasets_dir=args.datasets_dir, age_bin=parse_age_bins(args.age_bin), seq_len=args.seq_len,
                       batch_size=args.batch_size, num_workers=args.num_workers)
    loader = LoadDatasetCachedFirst(params, mode=args.data_source)
    dl = loader.get_data_loader()
    args.data_source_used = loader.source
    return dl['train'], dl['val'], dl['test']


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', choices=['gated', 'full_pe'], default='gated')
    ap.add_argument('--no_new_branch', action='store_true', help='control arms: original architecture')
    ap.add_argument('--kernel_size', type=int, default=99)
    ap.add_argument('--gate_init', type=float, default=0.0)
    ap.add_argument('--stage1_epochs', type=int, default=8)
    ap.add_argument('--stage2_epochs', type=int, default=20)
    # placeholder LRs: set stage-2 LRs relative to the LR used for your baseline finetune
    ap.add_argument('--lr_new1', type=float, default=1e-3)
    ap.add_argument('--lr_pe1', type=float, default=1e-4)
    ap.add_argument('--lr_new2', type=float, default=1e-4)
    ap.add_argument('--lr_pe2', type=float, default=1e-5)
    ap.add_argument('--lr_rest2', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-2)
    # stage-2 per-module LR multipliers (same meaning as in the baseline finetune script; 1.0 = flat LR)
    ap.add_argument('--seq_lr_mult', type=float, default=1.0)
    ap.add_argument('--head_lr_mult', type=float, default=1.0)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--val_metric', choices=['kappa', 'f1', 'loss'], default='kappa')
    ap.add_argument('--num_of_classes', type=int, default=5)
    # data (placeholders: copy batch_size / num_workers from your baseline finetune command)
    ap.add_argument('--datasets_dir', default=None, help='index parquet; only needed for live loading')
    ap.add_argument('--age_bin', default='all',
                    help="'6-12y' (one bin) | '3-5y,6-12y' (aggregate selected bins) | 'all' (aggregate all five) | "
                         "'each' (one separate run per bin; put {age_bin} in --baseline_ckpt and --out_dir)")
    ap.add_argument('--seq_len', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--data_source', choices=['auto', 'cache', 'live'], default='auto',
                    help="auto = preprocessed cache if usable else on-the-fly")
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--amp', action='store_true', help='bf16 autocast')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verify', action='store_true', help='only check gate=0 reproduces the baseline, then exit')
    return ap.parse_args()


def model_param(args):
    return Namespace(use_pretrained_weights=False, cuda=args.cuda, foundation_dir=None,
                     num_of_classes=args.num_of_classes,
                     ms_use_new_branch=not args.no_new_branch,
                     ms_kernel_size=args.kernel_size, ms_gate_init=args.gate_init)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def verify(args, device):
    """Baseline Model vs ModelMS (same checkpoint): logits must match when gate == 0."""
    sd = torch.load(args.baseline_ckpt, map_location='cpu')
    base = BaselineModel(Namespace(use_pretrained_weights=False, cuda=args.cuda, foundation_dir=None,
                                   num_of_classes=args.num_of_classes))
    ms = ModelMS(model_param(args))
    if isinstance(sd, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    base.load_state_dict(sd)
    load_baseline(ms, args.baseline_ckpt)
    base.to(device).eval()
    ms.to(device).eval()
    x = torch.randn(2, 20, 6, 6000, device=device)
    diff = (base(x) - ms(x)).abs().max().item()
    print(f'[verify] max |logit diff| baseline vs ModelMS = {diff:.3e} (gate_init={args.gate_init})')
    if args.gate_init == 0.0 or args.no_new_branch:
        assert diff < 1e-4, 'ModelMS does not reproduce the baseline - do NOT start training'
        print('[verify] OK: safe to start from this checkpoint.')


def score_of(m, metric):
    return -m['loss'] if metric == 'loss' else m[metric]


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, amp):
    model.eval()
    tot, n, ps, ys = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        c = logits.shape[-1]
        loss = loss_fn(logits.reshape(-1, c).float(), y.reshape(-1))
        tot += loss.item() * y.numel()
        n += y.numel()
        ps.append(logits.argmax(-1).reshape(-1).cpu())
        ys.append(y.reshape(-1).cpu())
    p, t = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    return {'loss': tot / n, 'acc': float(accuracy_score(t, p)),
            'f1': float(f1_score(t, p, average='macro')), 'kappa': float(cohen_kappa_score(t, p)),
            'per_class_f1': f1_score(t, p, average=None).tolist(),
            'confusion': confusion_matrix(t, p).tolist()}


def train_stage(model, stage, args, loaders, device, loss_fn, log):
    train_loader, val_loader = loaders[0], loaders[1]
    frozen = set_stage(model, stage, args.mode)
    if stage == 1:
        lrs = {'new': args.lr_new1, 'pe_old': args.lr_pe1}
        epochs = args.stage1_epochs
    else:
        lrs = {'new': args.lr_new2, 'pe_old': args.lr_pe2, 'rest': args.lr_rest2}
        epochs = args.stage2_epochs
    opt = make_optimizer(model, lrs, args.weight_decay,
                         mults={'seq': args.seq_lr_mult, 'head': args.head_lr_mult})
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'--- stage {stage} | mode={args.mode} | trainable params={n_tr:,} | epochs={epochs}')

    # epoch 0 (= starting point) is a candidate, so the selected model is never worse on val
    best = evaluate(model, val_loader, device, loss_fn, args.amp)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    log(f'stage {stage} ep 0 | val {fmt(best)}')
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        for m in frozen:
            m.eval()
        run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            run += loss.item()
        sched.step()
        val = evaluate(model, val_loader, device, loss_fn, args.amp)
        gate = getattr(model.backbone.patch_embedding, 'gate', None)
        g = f' gate={gate.item():.4f}' if gate is not None else ''
        log(f'stage {stage} ep {ep} | train loss {run / len(train_loader):.4f} | val {fmt(val)}{g}')
        if score_of(val, args.val_metric) > score_of(best, args.val_metric):
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f'early stop at epoch {ep}')
                break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(args.out_dir, f'stage{stage}_best.pt'))
    return best


def fmt(m):
    return f"loss {m['loss']:.4f} acc {m['acc']:.4f} mF1 {m['f1']:.4f} kappa {m['kappa']:.4f}"


def run(args):
    device = torch.device(f'cuda:{args.cuda}')
    seed_all(args.seed)
    if args.verify:
        verify(args, device)
        return

    base_dir = os.path.dirname(os.path.abspath(args.baseline_ckpt))
    out = os.path.abspath(args.out_dir)
    assert out != base_dir, 'out_dir must differ from the baseline checkpoint directory'
    assert not (os.path.isdir(out) and os.listdir(out)), f'{out} is not empty; choose a new --out_dir'
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(json.dumps(vars(args)))
    train_loader, val_loader, test_loader = get_loaders(args)
    model = ModelMS(model_param(args))
    load_baseline(model, args.baseline_ckpt)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()  # add class weights here if your baseline used them

    log(f'data source used: {args.data_source_used}')
    results = {'data_source': args.data_source_used,
               'start': {'val': evaluate(model, val_loader, device, loss_fn, args.amp),
                         'test': evaluate(model, test_loader, device, loss_fn, args.amp)}}
    log(f"start (== baseline when gate=0) | test {fmt(results['start']['test'])}")
    loaders = (train_loader, val_loader)
    for stage, epochs in ((1, args.stage1_epochs), (2, args.stage2_epochs)):
        if epochs <= 0:
            continue
        val = train_stage(model, stage, args, loaders, device, loss_fn, log)
        test = evaluate(model, test_loader, device, loss_fn, args.amp)
        results[f'stage{stage}'] = {'val': val, 'test': test}
        log(f'stage {stage} selected | test {fmt(test)}')
        with open(os.path.join(out, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    log('done')


def main():
    args = parse()
    if args.age_bin.strip().lower() == 'each':
        assert '{age_bin}' in args.baseline_ckpt and '{age_bin}' in args.out_dir, \
            "--age_bin each needs '{age_bin}' in both --baseline_ckpt and --out_dir"
        for b in AGE_BINS:
            a = copy.copy(args)
            a.age_bin = b
            a.baseline_ckpt = args.baseline_ckpt.format(age_bin=b)
            a.out_dir = args.out_dir.format(age_bin=b)
            if not os.path.isfile(a.baseline_ckpt):
                print(f'[each] SKIP {b}: baseline checkpoint not found: {a.baseline_ckpt}', flush=True)
                continue
            print(f'\n===== age bin {b} =====', flush=True)
            run(a)
    else:
        run(args)


if __name__ == '__main__':
    main()
