#!/bin/bash

############################ CIFAR experiments ############################
# Label noise levels
# c=0.2
c=0
datasets=CIFAR10
for model in resnet20
do

    CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model
    
    CUDA_VISIBLE_DEVICES=0 python3 -u train_pbfgs.py --epochs 20 --datasets $datasets --corrupt $c --params_start 0 --params_end 81  --batch-size 1024   --n_components 40 --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model 

    CUDA_VISIBLE_DEVICES=0 python3 -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81  --batch-size 128  --n_components 40 --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model 

done

############################ ImageNet experiments ##########################
c=0
datasets=ImageNet
for model in resnet18
do

    CUDA_VISIBLE_DEVICES=0,1,2,3  python3 main.py -a $model --epochs 90  --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/datasets/ILSVRC2012
    
    CUDA_VISIBLE_DEVICES=0,1,2 python3 -u train_pbfgs_imagenet.py --epochs 4 --print-freq 1000 --datasets $datasets --corrupt $c --alpha 0 --params_start 0 --params_end 241  --batch-size 256  --n_components 120 --arch=$model  --save-dir=save_$model |& tee -a log_$model 

done

############################ fine-tune experiments ##########################
# fine tune resnet18 0.005lr 15epoch
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py --arch resnet18 --pretrain --epochs 15 --lr 0.005 --dist-url 'tcp://127.0.0.1:9999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/datasets/ILSVRC2012

# P-BFGS 
CUDA_VISIBLE_DEVICES=0,1,2 python3 -u train_pbfgs_imagenet.py --epochs=1 --print-freq=100 --datasets=ImageNet --alpha=0 --params_start=1 --params_end=31 --batch-size=256 --n_components=30 --arch=resnet18 --save-dir=output/save_resnet18_0.005 &> res18_0005_5epoch_pbfgs.log

CUDA_VISIBLE_DEVICES=0,1,2 python3 -u train_pbfgs_imagenet.py --epochs=1 --print-freq=100 --datasets=ImageNet --alpha=0 --params_start=1 --params_end=61 --batch-size=256 --n_components=30 --arch=resnet18 --save-dir=output/save_resnet18_0.005 &> res18_0005_10epoch_pbfgs.log

CUDA_VISIBLE_DEVICES=0,1,2 python3 -u train_pbfgs_imagenet.py --epochs=1 --print-freq=100 --datasets=ImageNet --alpha=0 --params_start=1 --params_end=91 --batch-size=256 --n_components=30 --arch=resnet18 --save-dir=output/save_resnet18_0.005 &> res18_0005_15epoch_pbfgs.log



c=0
datasets=MNIST
for model in resnet20
do

    CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model

     

done






c=0.1
datasets=CIFAR10
n=3
for model in resnet20
do

    
    CUDA_VISIBLE_DEVICES=0 python3 -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128  --n_components $n --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model

    CUDA_VISIBLE_DEVICES=0 python3 -u train_rpsgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81  --batch-size 128  --n_components $n --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fpsgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128  --n_components $n --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model
 


done



c=0.4
datasets="CIFAR100"
n_values=(3 5 6 9 12 15 20 25)  
model="resnet32"

for n in "${n_values[@]}"  
do
    CUDA_VISIBLE_DEVICES=0 python3 -u train_SPCA.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fpsgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_STE.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS2.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}
done









c=0.01
datasets="CIFAR100"
 
model="resnet32"


do

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS2.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components 25 --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}
    

done








c=0.3
datasets="CIFAR100"
n_values=(3 5 6 9 12 15 20 25)  
model="resnet32"

for n in "${n_values[@]}"  
do
    CUDA_VISIBLE_DEVICES=0 python3 -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_rpsgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fpsgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}
done



c=0.3
datasets="CIFAR100"
n_values=(3 5 6 9 12 15 20 25)  
model="resnet32"

for n in "${n_values[@]}"  
do
    CUDA_VISIBLE_DEVICES=0 python3 -u train_fpsgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise$c\_$model |& tee -a log_${model}_n${n}
done



datasets=ImageNet
for model in resnet18
do

    CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model
    

done


c=0.05
datasets="CIFAR100"
model="resnet32"

# Initial Training
CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150 --save-dir=save_labelnoise$c\_$model |& tee -a log_$model

# Low-Dimensional Parameter Space Training
c=0.2
datasets="MNIST"
n_values=(3 5 6 9 )
model="resnet18"
for n in "${n_values[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 -u train_psgd.py --epochs 40 --datasets $datasets --lr 0.001 --corrupt $c --params_start 1 --params_end 10 --batch-size 128 --n_components $n --arch=$model --save-dir=MNIST_labelnoise{c} -a log_${datasets}_n${c}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_rpsgd.py --epochs 40 --datasets $datasets --lr 0.001 --corrupt $c --params_start 1 --params_end 10 --batch-size 128 --n_components $n --arch=$model --save-dir=MNIST_labelnoise{c} -a log_${datasets}_n${c}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fpsgd.py --epochs 40 --datasets $datasets --lr 0.001 --corrupt $c --params_start 1 --params_end 10 --batch-size 128 --n_components $n --arch=$model --save-dir=MNIST_labelnoise{c} -a log_${datasets}_n${c}
done



