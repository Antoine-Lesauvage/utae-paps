#!/bin/bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_vine_orchard.py \
    --dataset_folder "/home/onyxia/work/zenodo_data/PASTIS" \
    --res_dir "./results_tempo_augment_focal" \
    --pretrained_fold 3 \
    --vine_orchard_lr 0.0003 \
    --epochs 70 \
    --batch_size 4 \
    --fold 3 \
    --warmup 0 \
    --min_confidence 0.05 \
    --mask_threshold 0.25 \
    --min_remain 0.3 \
    --val_every 3 \
    --val_after 1 \
    --use_focal_loss \
    --focal_gamma 3.0 \
    --focal_alpha "1.16,29.78,250.42,10.11" \
    --temporal_aug \
    --aug_factor 3
