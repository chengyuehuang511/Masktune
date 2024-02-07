#!/bin/bash
cd /nethome/chuang475/flash/projects/Masktune

/nethome/chuang475/miniconda3/envs/ftp/bin/python3.8 -m src.main \
                    --dataset waterbirds \
                    --arch resnet50 \
                    --base_dir /nethome/chuang475/flash/ \
                    --lr 0.001 \
                    --use_cuda \
                    --optimizer sgd \
                    --train_batch 128 \
                    --test_batch 128 \
                    --masking_batch 128 \
                    --epochs 300 \
                    --use_pretrained_weights \
                    --dataset_dir /nethome/chuang475/flash/datasets/Waterbirds/raw \
                    --best_erm_model_checkpoint_path /nethome/chuang475/flash/runs/7/checkpoints/best_erm_model_checkpoint.pt \
                    --print_detail

# b /nethome/chuang475/flash/projects/Masktune/src/methods/base_method.py:589