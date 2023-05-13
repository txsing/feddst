gpu=$1
tag=$2
r=0.5

for sp in 0.0 0.5 0.6 0.8
do
for mu in 0.0 1.0
do
for dir in 0.0 0.001 0.005 0.01 0.05
do

echo EXP-mnist-iid-S${sp}-MU${mu}-D${dir}
python dst.py \
 --dataset mnist \
 --distribution iid \
 --lr 0.01 \
 -K 400 -C 20 \
 -R 400 -E 10 \
 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio ${r} \
 --pruning-begin 9 --pruning-interval 10 \
 -d $dir \
 --prox ${mu} \
 -o ${tag}mnist-iid-K400C20-E10R400-prox${mu}-s${sp}r${r}-dir${dir} --device $gpu

done
done
done
