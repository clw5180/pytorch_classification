# 20210110
#python train_holychen.py
#python train.py --model_name efficientnet-b3 --epochs 15 --lr 0.1 --accum_iter 1 --step_milestones 5 10 14
#python train.py --model_name efficientnet-b3 --epochs 15 --lr 0.1 --step_gamma 0.2 --accum_iter 1 --step_milestones 6 10 14



python train.py --model_name efficientnet-b4
python train.py --model_name se_resnext50_32x4d  --lr 0.02
python train.py --model_name resnext50_32x4d  --lr 0.02

#python train.py --model_name efficientnet-b3 --epochs 20 --lr 0.1 --step_gamma 0.3 --accum_iter 1 --step_milestones 7 11 15 18
#python train.py --model_name efficientnet-b4 --epochs 20 --lr 0.1 --step_gamma 0.3 --accum_iter 1 --step_milestones 7 11 15 18
#python train.py --model_name se_resnext50_32x4d --epochs 20 --lr 0.1 --step_gamma 0.3 --accum_iter 1 --step_milestones 7 11 15 18
#python train.py --model_name se_resnext101_32x4d --epochs 20 --lr 0.1 --step_gamma 0.3 --accum_iter 1 --step_milestones 7 11 15 18

#python train.py --model_name efficientnet-b4 --epochs 20 --lr 0.1
#python train.py --model_name efficientnet-b5 --epochs 20 --lr 0.1
#python train.py --model_name vit_base_patch16_384 --epochs 20 --lr 0.01
#python train.py --model_name se_resnext50_32x4d --epochs 20 --lr 0.1

#python train.py --model_name efficientnet-b3
#python train.py --model_name efficientnet-b4
#python train.py --model_name efficientnet-b5
#python train.py --model_name vit_base_patch16_384
#python train.py --model_name se_resnext50_32x4d
