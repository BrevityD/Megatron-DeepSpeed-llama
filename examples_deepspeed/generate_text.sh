#!/bin/bash
export TORCH_CUDA_ARCH_LIST=8.0
CHECKPOINT_PATH=/home/G01-A100-20240605/dwc/checkpoints/llama_tok21B_lr3e-4_min1e-6_w210M_d21B_cosine_gbs48_mbs3_g16_z1_mp2_pp2_seed42/global_step53406_merged_model/iter_0053406/mp_rank_00
tokenizer_id=/home/G01-A100-20240605/dwc/tokenizers/arxiv_vs256k_msl20.model
VOCAB_FILE=/home/G01-A100-20240605/dwc/tokenizers/arxiv_vs256k_msl20.vocab

b=1
mp=2
nodes=1
gpus=8

use_tutel=""
#use_tutel="--use-tutel"


ds_inference=""
#ds_inference="--ds-inference"

export CUDA_DEVICE_MAX_CONNECTIONS=1

# launch_cmd="deepspeed --num_nodes $nodes --num_gpus $gpus --master_port 29501 --include localhost:2"
launch_cmd="deepspeed --master_port 29501 --include localhost:2"
L=6
H=4096
A=32
#experts1=${experts[$k]}
program_cmd="tools/generate_samples_gpt.py \
       --tensor-model-parallel-size $mp \
       --num-layers $L \
       --hidden-size $H \
       --ffn-hidden-size 14336 \
       --num-attention-heads $A \
       --num-key-value-heads 8 \
       --max-position-embeddings 8192 \
       --tokenizer-type SentencePieceTokenizer \
       --bf16 \
       --mlp-type standard \
       --micro-batch-size $b \
       --seq-length 8192 \
       --out-seq-length 128 \
       --temperature 0.01 \
       --vocab-file $VOCAB_FILE \
       --tokenizer-model $tokenizer_id \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --log-interval 1 \
       --use-rotary-position-embeddings \
       --num-samples 0 \
       --load $CHECKPOINT_PATH \
       --make-vocab-size-divisible-by 128 \
       --recompute \
       $use_tutel $ds_inference"

echo $launch_cmd $program_cmd
$launch_cmd $program_cmd
