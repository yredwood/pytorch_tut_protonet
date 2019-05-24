python cifar_main.py \
    --num_gpus 4 --per_gpu_batch 64 \
    --max_epoch 220 --datatype cifar10 \
    --lr_decay 0.2 --lr_init 0.01 --lr_decay_list 60,120,160 \
    --aug_mixup 1 --arch wdres --wd 5e-4 --pr 0
