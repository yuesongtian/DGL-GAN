#!/bin/bash
python3.6 train_compression.py \
--shuffle --batch_size 100 --parallel \
--num_G_accumulations 1 --num_D_accumulations 2 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_ch 48 --D_ch 48 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 2500 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--data_root /apdcephfs/share_1367250/yuesongtian/ft_local/cifar-10-batches-py \
--use_multiepoch_sampler \
--pretrain_path /apdcephfs/share_1367250/yuesongtian/BigGAN_results/weights/hinge_ch96_cifar10_common/D_best0.pth \
--model BigGAN \
--experiment_name ch96_cifar10_base_G1-2_D1-2_iter25000 \
--g_loss loss_hinge_genDual --d_loss loss_hinge_dis \
--num_epochs 2000 \
--resume \


#--load_weights best4 \
#--G_ch 24 --D_ch 24 \
