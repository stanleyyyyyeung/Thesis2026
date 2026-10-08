"""
Multi-scale patch-encoder variant of the EEGMamba sleep-staging model.

Put this file in models/, NEXT TO eegmamba.py and model_for_isruc.py (same package). It imports from
them but never modifies them, so running/queued jobs that use the originals are
unaffected. Checkpoints written by this code use new file names only.

Contents
--------
PatchEmbeddingMS : PatchEmbedding + optional gated second time-scale branch
ModelMS          : Model (unchanged forward) with PatchEmbeddingMS swapped in
load_baseline    : load an existing finetuned Model checkpoint (strict about keys)
param_groups / set_stage / make_optimizer : freezing + optimizer helpers
"""
import copy

import torch
import torch.nn as nn
from einops import rearrange

from .eegmamba import PatchEmbedding
from .model_for_isruc import Model

NEW_PREFIX = ('backbone.patch_embedding.proj_in_b', 'backbone.patch_embedding.gate')
PE_PREFIX = 'backbone.patch_embedding.'


class PatchEmbeddingMS(PatchEmbedding):
    """Original PatchEmbedding plus a second time-domain branch.

    time_emb = proj_in(x) + gate * proj_in_b(x)

    With gate == 0 the output is identical to the original PatchEmbedding.
    Any odd kernel size k with padding k//2 and stride 25 keeps the output width
    at 8 for 200-sample patches, so 25 x 8 = 200 = d_model is preserved.
    Set use_new_branch=False to get the original architecture (control arm).
    """

    def __init__(self, in_dim, out_dim, d_model, seq_len,
                 use_new_branch=True, kernel_size=99, gate_init=0.0):
        super().__init__(in_dim, out_dim, d_model, seq_len)
        self.use_new_branch = use_new_branch
        if use_new_branch:
            assert kernel_size % 2 == 1, 'kernel_size must be odd'
            self.proj_in_b = nn.Sequential(
                nn.Conv2d(1, 25, kernel_size=(1, kernel_size), stride=(1, 25),
                          padding=(0, kernel_size // 2), bias=False),
                nn.GroupNorm(5, 25),
                nn.GELU(),
            )
            self.gate = nn.Parameter(torch.full((1,), float(gate_init)))
        self.track = False
        self.last_ratio = None

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = rearrange(mask_x, 'b c l d -> b d c l')
        time_x = rearrange(mask_x, 'b d c l -> b (c l) d').unsqueeze(1)

        time_emb = self.proj_in(time_x)
        if self.use_new_branch:
            branch = self.gate * self.proj_in_b(time_x)
            if self.track:
                self.last_ratio = (branch.detach().float().norm()
                                   / time_emb.detach().float().norm().clamp_min(1e-8)).item()
            time_emb = time_emb + branch
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        freq_x = rearrange(mask_x, 'b d c l -> b c l d')
        spectral = torch.fft.rfft(freq_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = time_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        return patch_emb + positional_embedding


class ModelMS(Model):
    """Same as Model (forward is inherited) but with PatchEmbeddingMS.

    Optional attributes on `param` (defaults in brackets):
      ms_use_new_branch [True], ms_kernel_size [99], ms_gate_init [0.0]
    Foundation weights are never loaded here; use load_baseline() instead.
    """

    def __init__(self, param):
        p = copy.copy(param)
        p.use_pretrained_weights = False
        super().__init__(p)
        old = self.backbone.patch_embedding
        new = PatchEmbeddingMS(
            in_dim=200, out_dim=200, d_model=200, seq_len=30,
            use_new_branch=getattr(param, 'ms_use_new_branch', True),
            kernel_size=getattr(param, 'ms_kernel_size', 99),
            gate_init=getattr(param, 'ms_gate_init', 0.0),
        )
        new.load_state_dict(old.state_dict(), strict=False)
        self.backbone.patch_embedding = new
        # parameters that are frozen by design (e.g. mask_encoding) must stay frozen
        self.never_train = {n for n, q in self.named_parameters() if not q.requires_grad}


def load_baseline(model, path):
    """Load a finetuned baseline checkpoint into a ModelMS.

    Only the new branch / gate may be missing; any other missing or unexpected
    key raises, so a silent partial load cannot happen.
    """
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(NEW_PREFIX)]
    if bad or unexpected:
        raise RuntimeError(f'Checkpoint mismatch. missing={bad[:10]} unexpected={list(unexpected)[:10]}')
    print(f'[load_baseline] loaded {path}; new (randomly initialised) keys: {len(missing)}')
    return model


def param_groups(model):
    """Split trainable-by-design params: new branch / old patch encoder / everything else."""
    g = {'new': [], 'pe_old': [], 'rest': []}
    for n, p in model.named_parameters():
        if n in model.never_train:
            continue
        if n.startswith(NEW_PREFIX):
            g['new'].append(p)
        elif n.startswith(PE_PREFIX):
            g['pe_old'].append(p)
        else:
            g['rest'].append(p)
    return g


def set_stage(model, stage, mode):
    """Set requires_grad for a stage and return modules that should be kept in eval().

    stage 1, mode 'gated'   : only new branch + gate train          (default)
    stage 1, mode 'full_pe' : whole patch encoder (old + new) trains
    stage 2 (any mode)      : everything trains
    """
    g = param_groups(model)
    if stage == 2:
        train_keys = ['new', 'pe_old', 'rest']
    elif mode == 'gated':
        train_keys = ['new']
    elif mode == 'full_pe':
        train_keys = ['new', 'pe_old']
    else:
        raise ValueError(f'unknown mode {mode}')
    if sum(len(g[k]) for k in train_keys) == 0:
        raise ValueError("No trainable parameters (mode='gated' needs the new branch; "
                         "use mode='full_pe' together with --no_new_branch for the control arm).")

    for p in model.parameters():
        p.requires_grad = False
    for k in train_keys:
        for p in g[k]:
            p.requires_grad = True

    frozen_eval = []
    if stage == 1:  # keep frozen parts deterministic (dropout off) so gate=0 == baseline
        pe = model.backbone.patch_embedding
        frozen_eval += [model.backbone.encoder, model.head, model.sequence_encoder, model.classifier]
        if mode == 'gated':
            frozen_eval += [pe.proj_in, pe.spectral_proj, pe.positional_encoding]
    return frozen_eval


def make_optimizer(model, lrs, weight_decay=1e-2, mults=None):
    """lrs: dict with keys 'new', 'pe_old', 'rest' (missing/None = not optimised).

    mults (optional, stage 2): multipliers applied to lrs['rest'] per module, mirroring the
    baseline finetune recipe: {'seq': seq_lr_mult, 'head': head_lr_mult}.
      sequence_encoder.*          -> lr * mults['seq']
      head.* and classifier.*     -> lr * mults['head']   (ASSUMED to match your trainer)
      backbone encoder (rest)     -> lr * 1
    """
    mults = mults or {}
    g = param_groups(model)
    names = {id(p): n for n, p in model.named_parameters()}
    pe = model.backbone.patch_embedding
    gate = [pe.gate] if getattr(pe, 'use_new_branch', False) else []
    gate_ids = {id(p) for p in gate}
    groups = []
    for k in ('new', 'pe_old'):
        params = [p for p in g[k] if p.requires_grad and id(p) not in gate_ids]
        if params and lrs.get(k):
            groups.append({'params': params, 'lr': lrs[k], 'weight_decay': weight_decay})
    if lrs.get('rest'):
        sub = {'backbone': [], 'seq': [], 'head': []}
        for p in g['rest']:
            if not p.requires_grad:
                continue
            n = names[id(p)]
            if n.startswith('sequence_encoder'):
                sub['seq'].append(p)
            elif n.startswith(('head', 'classifier')):
                sub['head'].append(p)
            else:
                sub['backbone'].append(p)
        m = {'backbone': 1.0, 'seq': mults.get('seq', 1.0), 'head': mults.get('head', 1.0)}
        for k, params in sub.items():
            if params:
                groups.append({'params': params, 'lr': lrs['rest'] * m[k], 'weight_decay': weight_decay})
    if gate and gate[0].requires_grad and lrs.get('new'):
        groups.append({'params': gate, 'lr': lrs['new'], 'weight_decay': 0.0})  # no decay on gate
    return torch.optim.AdamW(groups)
