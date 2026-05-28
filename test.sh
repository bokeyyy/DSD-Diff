#!/bin/bash

# 仅测试 Set14
benchmark="Set14"

# 论文 Table 3 对应的 3 种噪声等级
noises=(0 5 10)

# 定义 9 种各向异性高斯核的参数: lambda_1, lambda_2, theta
# 格式为: "l1,l2,th"
kernels=(
    "4.0,0.4,0"    # Kernel 1: 水平极细
    "4.0,0.4,30"   # Kernel 2: 轻微倾斜细长
    "4.0,0.4,45"   # Kernel 3: 45度倾斜细长
    "4.0,1.5,45"   # Kernel 4: 45度倾斜较胖
    "4.0,0.4,90"   # Kernel 5: 垂直极细
    "4.0,0.4,135"  # Kernel 6: 135度倾斜细长
    "4.0,1.5,135"  # Kernel 7: 135度倾斜较胖
    "3.0,2.0,135"  # Kernel 8: 略倾斜胖椭圆
    "3.0,3.0,0"    # Kernel 9: 正圆形 (各向同性)
)

for noise in "${noises[@]}"
do
    
    kernel_idx=1
    for params in "${kernels[@]}"
    do
        # 解析参数
        IFS=',' read l1 l2 th <<< "${params}"
        
        echo ">>> Running Kernel $kernel_idx (lambda_1=$l1, lambda_2=$l2, theta=$th) <<<"
        
        # 注意：这里 blur_type 强制设为 aniso_gaussian，这样才能生效 l1, l2, th 参数
        CUDA_VISIBLE_DEVICES=2 python test.py --test_only \
               --dir_data='./data' \
               --data_test=$benchmark \
               --scale='4' \
               --resume=600 \
               --n_GPUs=1 \
               --blur_type='aniso_gaussian' \
               --lambda_1=$l1 \
               --lambda_2=$l2 \
               --theta=$th \
               --noise=$noise \
               --save_results True
               
        kernel_idx=$((kernel_idx+1))
    done
done