target=($1)
sp=$2
dir=$3
gpu=$4
label=$5

pacs=('photo' 'art_painting' 'cartoon' 'sketch')
srcdomains=($(comm -3 <(printf "%s\n" "${pacs[@]}" | sort) <(printf "%s\n" "${target[@]}" | sort) | sort -n))
sources=$(printf '%s ' "${srcdomains[@]}")

python dst.py \
 --dataset pacs \
 --source ${sources} --target $target \
 --lr 0.001 \
 --clients 3 \
 --rounds 30 --rounds-between-readjustments 10 \
 --sparsity $sp --readjustment-ratio 0.01 \
 --pruning-begin 1 --pruning-interval 1 -E 2 \
 -d $dir \
 -o pacs-Ts-res18-E2R40-feddst-s${sp}r0.01-dir${dir}-${label} --device $gpu
