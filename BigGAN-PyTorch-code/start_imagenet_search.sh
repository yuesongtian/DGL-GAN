#!/bin/bash
python3.6 search.py \
--dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 96 \
--num_G_accumulations 1 --num_D_accumulations 1 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B1 0.0 --D_B2 0.999 --G_B2 0.999 \
--G_arch_lr 3e-4 --G_arch_B1 0.5 --G_arch_B2 0.999 --arch_weight_decay 1e-3 \
--G_attn 0 --D_attn 0 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 --G_shared \
--G_eval_mode \
--model BigGAN_search \
--G_ch 48 --D_ch 48 \
--ema_start 1000 \
--test_every 500 --save_every 100 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--use_multiepoch_sampler \
--pretrain_path /apdcephfs/share_1367250/yuesongtian/BigGAN_results/logs/hinge_ch96_cifar10_common/D_best4.pth \
--G_pretrain_path /apdcephfs/share_1367250/yuesongtian/BigGAN_results/logs/hinge_ch96_cifar10_common/G_ema_best4.pth \
--data_root /apdcephfs/share_1367250/yuesongtian/ \
--experiment_name search_48ch_imagenet_bat96_pretrainGbarDbar_woDeconv_woLeaInter \
--g_loss loss_hinge_gen --d_loss loss_hinge_dis \
--resume \
--num_epochs 10000 --inner_steps 20 --worst_steps 5 --outer_steps 20 \

#/apdcephfs/share_1367250/yuesongtian/pretrained_models/138k/G_ema.pth
#/apdcephfs/share_1367250/yuesongtian/pretrained_models/138k/D.pth
