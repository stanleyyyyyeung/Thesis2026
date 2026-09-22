import argparse
import random

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, \
    modma_dataset, nch_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a, model_for_modma


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Expected a boolean value (true/false), got {v!r}"
        )


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=6, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a, MODMA, NCH]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/Faced/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='/data/wjq/models_weights/Big/BigFaced', help='model_dir')
    parser.add_argument('--age_bin', type=str, default=None, help='NCH only: one of [1-2y, 3-5y, 6-12y, 13-18y, 19-100y]. ' 'Required when --downstream_dataset NCH.')
    parser.add_argument('--seq_len', type=int, default=20,
                        help='NCH only: number of 30s epochs per window (context length). '
                             'Must match the --seq-len the target parquet was built with '
                             '(nch_index_..._seqlen{N}.parquet) — NCHIndexDataset asserts '
                             'this against the parquet file metadata and will raise at '
                             'construction time if they disagree.')

    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=str2bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--head_lr_mult', type=float, default=5.0,
                        help='LR multiplier (relative to --lr) for the classifier/head params '
                             'when --multi_lr is set')
    parser.add_argument('--seq_lr_mult', type=float, default=2.0,
                        help='LR multiplier (relative to --lr) for sequence_encoder params '
                             'when --multi_lr is set. Kept separate from --head_lr_mult since '
                             'sequence_encoder is the module responsible for cross-epoch '
                             'coherence and tends to need a more conservative LR than the '
                             'classifier head to avoid converging to a per-epoch shortcut.')
    parser.add_argument('--frozen', type=str2bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=str2bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_EEGMamba.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        for i in range(1, 100):
            print('The {}th fold'.format(i))
            load_dataset = physio_dataset.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
            model = model_for_physio.Model(params)
            t = Trainer(params, data_loader, model)
            t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'NCH':
        assert params.age_bin is not None, (
            "--age_bin is required for NCH, e.g. --age_bin 6-12y"
        )
        assert params.seq_len is not None, (
            "--seq_len is required for NCH, e.g. --seq_len 100 "
            "(must match an existing nch_index_..._seqlen{N}.parquet)"
        )
        load_dataset = nch_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_mumtaz.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedvig.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        for i in range(100):
            print('The {}th time training'.format(i))
            load_dataset = tuev_dataset.LoadDataset(params)
            data_loader = load_dataset.get_data_loader()
            model = model_for_tuev.Model(params)
            t = Trainer(params, data_loader, model)
            t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        load_dataset = bciciv2a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_bciciv2a.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'MODMA':
        load_dataset = modma_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_modma.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
