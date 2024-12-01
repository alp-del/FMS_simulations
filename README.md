# Robust PCA for Low-Dimensional DNN Training


1. **PSGD**: Low Dimensional Trajectory Hypothesis is True: DNNs can be Trained in Tiny Subspaces (TPAMI 2022)
2. **FMS** : Fast, robust and non-convex subspace recovery
3. **TME**


## Requirements

To set up the environment, install the dependencies listed in `requirements.txt`.

### Environment Specifications

- **Python**: 3.6
- **PyTorch**: 1.4.0
- **CUDA**: 10.0.130

## CIFAR10 Experiments

The CIFAR10 experiments involve training a ResNet architecture (`resnet20`.`resnet32`) using SGD for 150 epochs. Once the initial training completes, further fine-tuning is conducted in a reduced parameter space using PSGD, RPSGD, and FPSGD.

### Training Steps

1. **Initial Training**: Train a neural network using SGD to establish a parameter space.


```bash
c=0  # Level of corruption
datasets="CIFAR10"
model="resnet20"
CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150 --save-dir=save_labelnoise${c}_${model} |& tee -a log_$model
```

2. **Low-Dimensional Training**: Refine the model within a low-dimensional parameter space using projected gradient methods. The **params_start** and **params_end** are the parameters to select the parameter space. And **n_values** are the reduced dimensions. **c** is the corruption level.

```bash
c=0  # Level of corruption
datasets="CIFAR10"
model="resnet20"
n_values=(3 5 6 9 12 15 20 25)

for n in "${n_values[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SPCA.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fms.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fms2.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_tme.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_STE.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS2.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}
done
```

For CIFAR100, we use a similar process but with a different model (resnet32)
1. **SGD Training**
```bash
c=0.05
datasets="CIFAR100"
model="resnet32"
CUDA_VISIBLE_DEVICES=0 python3 -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model
```
2. **Low-Dimensional Parameter Space Training**
```bash
c=0.05
datasets="CIFAR100"
model="resnet32"
n_values=(3 5 6 9 12 15 20 25)
for n in "${n_values[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SPCA.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fms.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_fms2.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_tme.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_STE.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}

    CUDA_VISIBLE_DEVICES=0 python3 -u train_SFMS2.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81 --batch-size 128 --n_components $n --arch=$model --save-dir=save_labelnoise${c}_${model} |& tee -a log_${model}_n${n}
done
```
