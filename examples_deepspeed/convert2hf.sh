cd /home/G01-A100-20240605/dzj/Megatron-DeepSpeed/tools/convert_checkpoint/

checkpoint_dir=/home/G01-A100-20240605/dwc/checkpoints/llama_tok21B_lr3e-4_min1e-6_w210M_d21B_cosine_gbs48_mbs3_g16_z1_mp2_pp2_seed42/
# checkpoint_dir=${PATH_TO_CKPT}/${MODEL_SIGNATURE}/global_step${MODEL_STEP}

python3 convert_to_hf.py \
    --input_dir ${checkpoint_dir} \
    --output_dir ${checkpoint_dir}_hf \
    --no_save_tokenizer \
    --architecture "llama" \
    --config_file llama_config.json \