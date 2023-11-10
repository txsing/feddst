target=($1)
sp=$2
dir=$3
gpu=$4
label=$5
ratio=$6

pacs=('photo' 'art_painting' 'cartoon' 'sketch')
srcdomains=($(comm -3 <(printf "%s\n" "${pacs[@]}" | sort) <(printf "%s\n" "${target[@]}" | sort) | sort -n))
sources=$(printf '%s ' "${srcdomains[@]}")

epochs=2
rounds=40
#ratio=0.1
X=7
Y=0
Z=2

ologfile=pacs-T$target-res18-E${epochs}R${rounds}-${X}_${Y}_${Z}-feddst-s${sp}r${ratio}-dir${dir}-${label}
echo ${ologfile}

python dst.py \
 --dataset pacs \
 --source ${sources} --target $target \
 --lr 0.001 \
 --clients 3 \
 --rounds ${rounds} --rounds-between-readjustments $X \
 --sparsity $sp --readjustment-ratio ${ratio} \
 --pruning-begin $Y --pruning-interval $Z -E ${epochs} \
 -d $dir \
 -o ${ologfile} --device $gpu