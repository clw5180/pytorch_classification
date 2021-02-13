sleep 25m
python train_yerramvarun14.py

# 20210208  effb5, 456 bitemperedloss and effb4 512 TaylorLoss, cutmix_prob is all 1.0(by official version)
#sleep 15m
#python train_yerramvarun11.py
#wait
#python train_yerramvarun12.py

# 20210207  effb5, 456taylorloss, cutmix_prob is 1.0(by official version)
#sleep 1.5h
#python train_yerramvarun10.py


# 20210204   train_yerramvarun6: effb5, 456
#python train_yerramvarun7.py
#wait
#python train_yerramvarun8.py

## 20210204   train_yerramvarun6: effb5, 456
#python train_yerramvarun6.py
#wait
#python train_yerramvarun4.py

#sleep 4.7h
#python train_yerramvarun5.py  # effb4

#python train_holychen.py
#python train_holychen2.py

#
#sleep 5.5h
#python train_yerramvarun2.py


# 20210131
#python train_holychen.py --model_arch vit_base_patch16_384 --lr 0.01 --image_size 384  ## have bug!!
#wait
#python train_holychen.py --model_arch efficientnet-b3 --lr 0.1
#wait
#python train_holychen.py --model_arch se_resnext50_pretrainedmodels --lr 0.02
#wait
#python train_holychen.py --model_arch se_resnext50_timm --lr 0.02
#wait


#python train.py --model_name efficientnet-b3 --epochs 20 --lr 0.1 --accum_iter 1 --step_milestones 12 16 19
#python train.py --model_name se_resnext50_32x4d --epochs 20 --lr 0.02 --accum_iter 1 --step_milestones 12 16 19
#python train.py --model_name vit_base_patch16_384 --epochs 20 --lr 0.01 --accum_iter 1 --step_milestones 12 16 19



# 20210110
#python train_holychen.py
#python train.py --model_name efficientnet-b3 --epochs 15 --lr 0.1 --accum_iter 1 --step_milestones 5 10 14
#python train.py --model_name efficientnet-b3 --epochs 15 --lr 0.1 --step_gamma 0.2 --accum_iter 1 --step_milestones 6 10 14



#python train.py --model_name efficientnet-b4
#python train.py --model_name se_resnext50_32x4d  --lr 0.02
#python train.py --model_name resnext50_32x4d  --lr 0.02

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
