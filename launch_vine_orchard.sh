#!/bin/bash
python train_vine_orchard.py \
    --dataset_folder "/home/onyxia/work/zenodo_data/PASTIS" \
    --res_dir "./results_vine_orchard" \
    --pretrained_fold 1 \
    --vine_orchard_lr 0.001 \
    --epochs 50 \
    --batch_size 4 \
    --fold 1 \
    --warmup 0 \
    --val_every 2 \
    --val_after 5