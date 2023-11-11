rounds=400; #R
epochs=5; #E
total_clients=100; #K
rd_clients=10; #C
sp=0.0; #S
rd_adjust=20; #T
mu=0.0; #m
dir=0.0; #d
gpu=0 #g

prune_beg=0;
prune_int=20;
alpha=0.1;
tag='None';
lr=0.001

eval set -- $(getopt -o R:,E:,K:,C:,S:,T:,m:,d:,g: -l pb:,pt:,alpha:,tag:,adam:,drill: -- "$@")

while true; do
  case "$1" in
    --pb) prune_beg=$2; shift 2 ;;
    --pt) prune_int=$2; shift 2 ;;
    --alpha) alpha=$2; shift 2 ;;
    --drill) drill=$2; shift 2 ;;
    --adam) adam=$2; shift 2 ;;
    --tag) tag=$2; shift 2 ;;
    -R) rounds=$2; shift 2 ;;
    -E) epochs=$2; shift 2 ;;
    -K) total_clients=$2; shift 2 ;;
    -C) rd_clients=$2; shift 2 ;;
    -S) sp=$2; shift 2 ;;
    -T) rd_adjust=$2; shift 2 ;;
    -m) mu=$2; shift 2 ;;
    -d) dir=$2; shift 2 ;;
    -g) gpu=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "Invalid option: $1"; exit 1 ;;
  esac
done

if [ -z "$drill" ]; then
  drill=false
fi
if [ -z "$adam" ]; then
  adam=false
fi

if [ "$adam" = "true" ] ; then
    echo 'Use Adam!'
    lr=0.0001
fi

logfile=mnist-noniid-Adam${adam}-E${epochs}R${rounds}K${total_clients}C${rd_clients}-prox${mu}-s${sp}r${alpha}-dir${dir}-${rd_adjust}_${prune_beg}_${prune_int}-${tag}
echo ${logfile}

python dst.py \
 --distribution noniid --batch-size 50 \
 --drill ${drill} \
 --adam ${adam} \
 --dataset mnist \
 --lr ${lr} \
 -K ${total_clients} -C ${rd_clients} \
 -R ${rounds} -E ${epochs} \
 --rounds-between-readjustments ${rd_adjust} \
 --sparsity $sp --readjustment-ratio ${alpha} \
 --pruning-begin ${prune_beg} --pruning-interval ${prune_int} \
 -d ${dir} \
 --prox ${mu} \
 -o ${logfile} --device $gpu