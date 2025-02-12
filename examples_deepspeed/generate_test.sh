tokenizer_id=/home/G01-A100-20240605/dwc/tokenizers/arxiv_vs256k_msl20.model
VOCAB_FILE=/home/G01-A100-20240605/dwc/tokenizers/arxiv_vs256k_msl20.vocab
MERGE_FILE=/home/G01-A100-20240605/dzj/Megatron-DeepSpeed/examples_deepspeed/gpt2-merges.txt
CHECKPOINT_PATH=/home/G01-A100-20240605/dwc/checkpoints/llama_tok21B_lr3e-4_min1e-6_w210M_d21B_cosine_gbs48_mbs3_g16_z1_mp2_pp2_seed42/global_step53406

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPT_ARGS=" \
    --num-layers 6 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --tokenizer-model $tokenizer_id \
    --micro-batch-size 1 \
    --fp16 \
    "

MAX_OUTPUT_SEQUENCE_LENGTH=1024
TEMPERATURE=1.0
TOP_P=0.9
NUMBER_OF_SAMPLES=2
OUTPUT_FILE=samples.json

python tools/generate_samples_gpt.py \
    $GPT_ARGS \
    --load $CHECKPOINT_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --out-seq-length $MAX_OUTPUT_SEQUENCE_LENGTH \
    --temperature $TEMPERATURE \
    --genfile $OUTPUT_FILE \
    --num-samples $NUMBER_OF_SAMPLES \
    --top_p $TOP_P \
    --recompute