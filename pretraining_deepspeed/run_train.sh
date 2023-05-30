#~/bin/bash

#1: number of GPUs
#2: Model File Address
#3: BertSquad Data Directory Address
#4: Output Directory Address

# NGPU_PER_NODE=$1
NGPU_PER_NODE=8
MASTER_PORT=${7:-29500}

# Force deepspeed to run with only local node
NUM_NODES=1
HOSTFILE=/dev/null

NGPU=$((NGPU_PER_NODE*NUM_NODES))

# config_json=pretraining_deepspeed/deepspeed_config.json
# config_json=deepspeed_onebitadam_bsz96_config.json
config_json=pretraining_deepspeed/deepspeed_config_OneBitAdam.json
# run_cmd="deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
#        --master_port=${MASTER_PORT} \
#        --hostfile ${HOSTFILE} \
#        pretraining_deepspeed/train_Parallel_v2.py \
#        --deepspeed \
#        --deepspeed_config ${config_json} \
#        "
#        # --deepspeed_transformer_kernel \

run_cmd="deepspeed \
       --master_port=${MASTER_PORT} \
       --hostfile ${HOSTFILE} \
       --include='localhost:0,1,2,3,4,5,6,7' pretraining_deepspeed/train_Parallel_v2.py \
       --deepspeed \
       --deepspeed_config ${config_json} \
       "
       # --deepspeed_transformer_kernel \
echo ${run_cmd}
eval ${run_cmd}

# nohup sh pretraining_deepspeed/run_train.sh > run_2023.log 2>&1 &
