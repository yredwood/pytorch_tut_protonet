python chexpert_train.py \
    --num_gpus 8 --per_gpu_batch 32 \
    --max_epoch 100 --datatype imagenet \
    --lr_decay 0.1 --lr_init 0.01 --lr_decay_list 50,80 \
    --aug_mixup 1 --arch vgg --wd 1e-4 --pr 0
