#!/bin/bash

# Load Modules
module purge
module load python/3.6.3
module load cuda/10.1
module load cudnn/7.6.5

# Activate Virtual Environment
source myenv/bin/activate

# Set CUDA Environment Variables
export CUDA_HOME=/common/software/install/migrated/cuda/10.1
export CUDNN_HOME=/common/software/install/migrated/cudnn/7.6.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$LD_LIBRARY_PATH

# Define Variables
datasets="CIFAR10"
model="resnet20"
corruption_levels=(0 0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.4)
n_values=( 3 6 9 12 15 20 25)

# Create Logs Directory if Not Exists
mkdir -p logs

# Main Training Loop
for c in "${corruption_levels[@]}"
do
    echo "Starting low-dimensional trainings for corruption level c=$c..."
    for n in "${n_values[@]}"
    do
        methods=("psgd" "FMS" "STE3")
        for method in "${methods[@]}"
        do
            echo "Running $method with n_components=${n} and c=${c}..."
            CUDA_VISIBLE_DEVICES=0 python3 -u train_${method}.py \
                --epochs 40 --datasets $datasets --lr 1 --corrupt $c \
                --params_start 0 --params_end 81 --batch-size 128 \
                --n_components $n --arch=$model \
                --save-dir=logs/save_labelnoise${c}_${model} |& tee -a logs/log_${method}.log
        done
    done
done

#for c in "${corruption_levels[@]}"
#do
#    echo "Starting initial training with SGD for corruption level c=$c..."
#    CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py \
#        --datasets $datasets \
#        --lr 0.1 \
#        --corrupt $c \
#        --arch=$model \
#        --epochs=150 \
#        --save-dir=logs/save_labelnoise${c}_${model} |& tee -a logs/log_SGD.log

#    echo "Starting low-dimensional trainings for corruption level c=$c..."
#    for n in "${n_values[@]}"
#    do
#        methods=("psgd" "FMS" "cTME")
#        for method in "${methods[@]}"
##        do
#            echo "Running $method with n_components=${n} and c=${c}..."
#            CUDA_VISIBLE_DEVICES=0 python3 -u train_${method}.py \
#                --epochs 40 --datasets $datasets --lr 0.01 --corrupt $c \
#                --params_start 0 --params_end 81 --batch-size 128 \
#                --n_components $n --arch=$model \
#                --save-dir=logs/save_labelnoise${c}_${model} |& tee -a logs/log_${method}.log
#        done
#    done
#done



echo "All runs completed."
