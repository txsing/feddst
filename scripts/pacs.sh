sp=$1
dir=$2
gpu=$3
label=$4
python dst.py \
 --dataset pacs \
 --source photo cartoon art_painting --target sketch \
 --lr 0.001 \
 --clients 3 \
 --rounds 30 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio 0.01 \
 --pruning-begin 1 --pruning-interval 1 -E 1 \
 -d $dir \
 -o pacs-Ts-res18-E2R40-feddst-s${sp}r0.01-dir${dir}-${label} --device $gpu
