# author: Brevity_D
# reference: GPT-NEOX
# https://github.com/EleutherAI/gpt-neox/blob/main/tools/ckpts/convert_neox_to_hf.py

import os
import sys

# import yaml
import json
import argparse
from tqdm import tqdm

import torch
from transformers import (
    MistralConfig,
    LlamaConfig,
    GPTNeoXConfig,
    AutoModelForCausalLM,
    AutoConfig,
)

from typing import List, Literal

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
) # 解决 torch save 中保留相对路径的问题。将缺失的 `module` 上级目录（\Megatron-DeepSpeed）加入到环境变量中

from megatron.tokenizer import build_tokenizer

# Model definitions: a list of keys, and where they fall in terms of handling them in the presence of TP.
# in format : {model arch: {param type: {param in neox: param in HF}}}

MODEL_KEYS = {
    "neox": {
        "COLUMN_PARALLEL_LINEAR_KEYS": {
            "mlp.dense_h_to_4h.weight": "mlp.dense_h_to_4h.weight",
            "mlp.dense_h_to_4h.bias": "mlp.dense_h_to_4h.bias",
            "attention.query_key_value.weight": "attention.query_key_value.weight",
            "attention.query_key_value.bias": "attention.query_key_value.bias",  # TODO: handle GQA separately?
        },
        "ROW_PARALLEL_LINEAR_KEYS": {
            "attention.dense.weight": "attention.dense.weight",
            "mlp.dense_4h_to_h.weight": "mlp.dense_4h_to_h.weight",
        },
        "ROW_PARALLEL_BIAS_KEYS": {
            "mlp.dense_4h_to_h.bias": "mlp.dense_4h_to_h.bias",
            "attention.dense.bias": "attention.dense.bias",
        },
        "NORM_KEYS": {
            "input_layernorm.weight": "input_layernorm.weight",
            "input_layernorm.bias": "input_layernorm.bias",
            "post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "post_attention_layernorm.bias": "post_attention_layernorm.bias",
        },
        "FINAL_NORM_KEYS": {
            "norm.weight": "weight",
            "norm.bias": "bias",
        },
    },
    "llama_fork": {
        "COLUMN_PARALLEL_LINEAR_KEYS": {
            "mlp.w1.weight": "mlp.gate_proj.weight",
            "mlp.w3.weight": "mlp.up_proj.weight",
        },
        "ROW_PARALLEL_LINEAR_KEYS": {
            "attention.dense.weight": "self_attn.o_proj.weight",
            "mlp.w2.weight": "mlp.down_proj.weight",
        },
        "ROW_PARALLEL_BIAS_KEYS": {},  # No biases in RowParallelLinear layers
        "NORM_KEYS": {
            "input_layernorm.scale": "input_layernorm.weight",
            "post_attention_layernorm.scale": "post_attention_layernorm.weight",
        },
        "FINAL_NORM_KEYS": {
            "norm.scale": "weight",
        },
        "GQA_QKV_KEYS": {  # because Llama can have Grouped Query Attention and has separate Q, K, and V linear proj params, handle them separately.
            "attention.query_key_value.weight": [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
        },
    },
    "llama": {
        "COLUMN_PARALLEL_LINEAR_KEYS": {
            # "mlp.w1.weight": "mlp.gate_proj.weight",# BD：这个真的找不到
            "mlp.dense_h_to_4h.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"], # BD: 反转了, 这里需要把 h to 4h 拆成两层
        },
        "ROW_PARALLEL_LINEAR_KEYS": {
            "self_attention.dense.weight": "self_attn.o_proj.weight",
            "mlp.dense_4h_to_h.weight": "mlp.down_proj.weight",
        },
        "ROW_PARALLEL_BIAS_KEYS": {},  # No biases in RowParallelLinear layers
        "NORM_KEYS": {
            "input_layernorm.weight": "input_layernorm.weight",
            "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        },
        "FINAL_NORM_KEYS": {
            "weight": "weight",
        },
        "GQA_QKV_KEYS": {  # because Llama can have Grouped Query Attention and has separate Q, K, and V linear proj params, handle them separately.
            "self_attention.query_key_value.weight": [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
        },
    },
}

MODEL_KEYS["mistral"] = MODEL_KEYS["llama"]


def load_partitions(
    input_checkpoint_path: str, mp_partitions: int, layer_idx: int, sequential: bool
) -> List[torch.Tensor]:
    """Returns a list containing all states from a model (across MP partitions)"""

    if sequential:
        filename_format = f"mp_rank_{{i:02}}_model_states.pt"
    else:
        filename_format = f"layer_{layer_idx:02}-model_{{i:02}}-model_states.pt"

    loaded_tp_ranks = [
        torch.load(
            os.path.join(
                input_checkpoint_path,
                filename_format.format(i=i),
            ),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        for i in range(mp_partitions)
    ]

    return loaded_tp_ranks


def get_state(
    state_dicts: List[torch.Tensor], key: str, layer_idx: int, sequential: bool
) -> torch.Tensor:
    """Helper that returns a list containing a given weight's state from each MP partition, for a given layer in the model."""

    if sequential:
        # use the correct key into the sequential dict for given weight/provided key
        key = f"sequential.{layer_idx}.{key}"

        return [state_dict["module"][key] for state_dict in state_dicts]
    else:
        # For the PipelineModule case, we don't need any key / module prefix. just grab this weight value.
        # layer_idx is also ignored because we've loaded only this layer's weights, ahead of time.
        key = key
        # if key == "lm_head.weight":
        #     print(f">> state_dicts: {state_dicts}")
        #     print(">> state_dicts_size: ", state_dicts[0][key].size()," and ",state_dicts[1][key].size())

        return [state_dict[key] for state_dict in state_dicts]


def get_key(loaded_config, key, default=None):
    """
    Search for a given key in a NeoX yaml. normalizes underscores -> hyphens
    """
    key = key.replace("_", "-")
    try:
        return loaded_config[key]
    except KeyError:
        key = key.replace("-", "_")
        try:
            return loaded_config[key]
        except KeyError:
            return default


def create_config(neox_config, architecture="neox"):
    """take in a loaded yaml from NeoX and assign relevant values to HF config.
    Returns: GPTNeoXConfig() object
    """

    def gated_size(hidden_dim):
        # takes in a hidden dim and calculates intermediate dim of a LLaMAParallelMLP.
        # (only used if intermediate_size not specified in config)
        # hidden-size * 8 / 3 , rounded up to nearest multiple of 256
        # BD: use 4*h
        ff_dim = int(3.5*hidden_dim) # 14336
        # ff_dim = int(2 * hidden_dim * 4 / 3)
        # ff_dim = 256 * ((ff_dim + 256 - 1) // 256)
        return ff_dim

    class TokenizerArgs:
        # kinda hacky.
        # this is to get something with the same interface as is used in build_tokenizer()
        # without diving into loading a neox_args object or using argparse etc.
        def __init__(self, neox_config):
            self.make_vocab_size_divisible_by = get_key(
                neox_config, "make-vocab-size-divisible-by", default=128
            )
            self.tensor_model_parallel_size = get_key(neox_config, "model-parallel-size")
            # self.vocab_file = get_key(neox_config, "vocab-file")
            self.tokenizer_model = get_key(neox_config, "tokenizer_model")
            # self.merge_file = get_key(neox_config, "merge-file")
            self.tokenizer_type = get_key(neox_config, "tokenizer-type")
            self.seq_length = get_key(neox_config, "seq_length")
            self.vocab_extra_ids = 0
            self.rank = 0

    args = TokenizerArgs(neox_config)
    tokenizer = build_tokenizer(args)
    print("*-"*10)
    print("tokenizer built")
    print("*-"*10)
    try:  # GPT2TokenizerFast raises NotImplementedError
        pad_token = tokenizer.pad
    except:
        pad_token = (
            1  # pad defaulting to 1. follows convention from GPT-NeoX-20b tokenizer
        )

    # TODO: change the default value here based on discussion regarding `gpt_j_tied` config parameter's default
    use_tied_lns = get_key(neox_config, "gpt-j-tied", False)

    if use_tied_lns:
        raise NotImplementedError(
            """ERROR: Huggingface Transformers does not yet support a single shared layernorm
                per transformer block for GPT-NeoX models trained  w/ GPT-J parallel residuals.
                See https://github.com/EleutherAI/gpt-neox/pull/481 for further details."""
        )

    # set all config values.

    # shared config parameters.
    args = {
        "vocab_size": args.padded_vocab_size,# 128256, #
        "hidden_size": get_key(neox_config, "hidden-size"),
        "num_hidden_layers": get_key(neox_config, "num-layers"),
        "num_attention_heads": get_key(neox_config, "num-attention-heads"),
        "max_position_embeddings": get_key(neox_config, "max-position-embeddings"),
        "initializer_range": get_key(neox_config, "init-method-std", 0.02),
        "tie_word_embeddings": (not get_key(neox_config, "no-weight-tying", False)),
        "use_cache": True,
    }

    # print(f">> tokenizer args: {args}")
    if architecture == "mistral" or architecture == "llama":
        args.update(
            {
                "intermediate_size": get_key(
                    neox_config,
                    "intermediate-size",
                    gated_size(get_key(neox_config, "hidden-size")),
                ),
                "num_key_value_heads": get_key(
                    neox_config,
                    "num-kv-heads",
                    get_key(neox_config, "num-attention-heads"),
                ),
                "hidden_act": get_key(neox_config, "activation", default="silu"),
                "rms_norm_eps": get_key(neox_config, "rms-norm-epsilon", 1.0e-6),
                "bos_token_id": tokenizer.eod,
                "eos_token_id": tokenizer.eod,
                "rope_theta": get_key(neox_config, "rotary-emb-base", 10000.0),
            }
        )

        if architecture == "mistral":
            # mistral-specific options
            args.update(
                {
                    "sliding_window": get_key(
                        neox_config, "sliding-window-width", 4096
                    ),
                }
            )
            hf_config = MistralConfig(**args)
        elif architecture == "llama":
            # llama-specific options
            args.update(
                {
                    # NeoX library defaults to using bias in attention
                    "attention_bias": get_key(
                        neox_config, "use_bias_in_attn_linear", True
                    ),
                }
            )
            hf_config = LlamaConfig(**args)
    else:
        # GPT-NeoX HF model class-specific options
        args.update(
            {
                "rotary_pct": get_key(neox_config, "rotary-pct", default=1.0),
                "rotary_emb_base": get_key(
                    neox_config, "rotary-emb-base", default=1000.0
                ),
                "use_parallel_residual": get_key(neox_config, "gpt-j-residual", False),
                "layer_norm_eps": get_key(neox_config, "layernorm-epsilon", 1e-5),
                "intermediate_size": get_key(
                    neox_config,
                    "intermediate-size",
                    4 * get_key(neox_config, "hidden-size"),
                ),
            }
        )
        hf_config = GPTNeoXConfig(**args)

    return hf_config


def reshard_and_split_qkv(
    param_mapping: dict,  # a dictionary mapping the QKV weight keys in GPT-NeoX -> a list of keys representing the Q, K, and V weight keys the HF model will use
    hf_config: AutoConfig,  # a HF model config for the model
    loaded_tp_ranks: List[torch.Tensor],
    layer_idx: int,
    sequential: bool,
):
    """
    A helper function which performs reshaping and sharding to make the QKV projection from NeoX compatible with HF Llama models,
    even when grouped-query attention is required.
    """
    for key, hf_keys in param_mapping.items():
        assert (
            isinstance(hf_keys, list) and len(hf_keys) == 3
        ), "Must map QKV to precisely 3 resulting weight matrices."

    # BD: 20240621, QGA
    # for key, hf_keys in param_mapping.items():
    #     # we first merge the QKV proj. across TP ranks
    #     # BD: for example, when TP=2, we get layer_02 with 2 .pt files:
    #     # layer_02-model_00-model_states.pt, layer_02-model_01-model_states.pt
    #     # so sharded_qkv = torch.stack([model_00, model_01])
    #     sharded_qkv = torch.stack(
    #         get_state(loaded_tp_ranks, key, layer_idx, sequential), dim=0
    #     )

    #     sharded_qkv = sharded_qkv.view(
    #         len(loaded_tp_ranks),
    #         hf_config.num_attention_heads // len(loaded_tp_ranks),
    #         int(
    #             hf_config.hidden_size//hf_config.num_attention_heads * 3
    #         ),
    #         hf_config.hidden_size,
    #     )  # is meant to convert to shape [TP_SIZE, NUM_QUERY_HEADS_PER_SHARD, dims_per_head * (1 + 2 * kv-to-q head ratio), hidden_size]
        
    #     q, k, v = torch.split(
    #             sharded_qkv,
    #             [
    #                 hf_config.hidden_size // hf_config.num_attention_heads,
    #                 hf_config.hidden_size // hf_config.num_attention_heads,
    #                 hf_config.hidden_size // hf_config.num_attention_heads,
    #             ],
    #             dim=2,
    #         )
    #         # splits along the (dims_per_head * (1 + 2 * kv-to-q head ratio)_ dim to get 3 tensors:
    #         # 3 x [TP_SIZE, NUM_Q_HEADS_PER_SHARD, dims_per_head, hidden_size]
    #         # these are the Q, and K, V tensors respectively.
    #     q, k, v = q.squeeze(dim=2), k.squeeze(dim=2), v.squeeze(dim=2)
    #     q = q.view(
    #         hf_config.num_attention_heads,
    #         hf_config.hidden_size // hf_config.num_attention_heads,
    #         hf_config.hidden_size,
    #     ).reshape(hf_config.hidden_size, hf_config.hidden_size)
    #     k = k.reshape(
    #         hf_config.num_attention_heads,
    #         hf_config.hidden_size // hf_config.num_attention_heads,
    #         hf_config.hidden_size,
    #     ).reshape(hf_config.hidden_size, hf_config.hidden_size)
    #     v = v.reshape(
    #         hf_config.num_attention_heads,
    #         hf_config.hidden_size // hf_config.num_attention_heads,
    #         hf_config.hidden_size,
    #     ).reshape(hf_config.hidden_size, hf_config.hidden_size)

    # BD: 24.6.21 trying to fix GQA
    for key, hf_keys in param_mapping.items():
        # we first merge the QKV proj. across TP ranks
        # BD: for example, when TP=2, we get layer_02 with 2 .pt files:
        # layer_02-model_00-model_states.pt, layer_02-model_01-model_states.pt
        # so sharded_qkv = torch.stack([model_00, model_01])
        sharded_qkv = torch.stack(
            get_state(loaded_tp_ranks, key, layer_idx, sequential), dim=0
        )
        # should now have shape [TP_SIZE, (hidden_size + 2 * kv_hidden_size) / TP_SIZE, hidden_size].
        # print(f">> should have shape [TP_SIZE={len(loaded_tp_ranks)}, (hidden_size={hf_config.hidden_size} + 2 * kv_hidden_size={hf_config.num_key_value_heads / hf_config.num_attention_heads}) / TP_SIZE, hidden_size] and actually get: {sharded_qkv.size()}") # BD

        # print(f">> trying to view to shape [{len(loaded_tp_ranks)}, {hf_config.num_attention_heads // len(loaded_tp_ranks)}, {int(hf_config.hidden_size//hf_config.num_attention_heads*(1+2*hf_config.num_key_value_heads/hf_config.num_attention_heads))}, {hf_config.hidden_size}]") # BD

        sharded_qkv = sharded_qkv.view(
            len(loaded_tp_ranks),
            hf_config.num_attention_heads // len(loaded_tp_ranks),
            int(
                hf_config.hidden_size
                // hf_config.num_attention_heads
                * (
                    1
                    + 2 * hf_config.num_key_value_heads / hf_config.num_attention_heads # kv 头个数
                )
            ),
            hf_config.hidden_size,
        )  # is meant to convert to shape [TP_SIZE, NUM_QUERY_HEADS_PER_SHARD, dims_per_head * (1 + 2 * kv-to-q head ratio), hidden_size]

        q, k, v = torch.split(
            sharded_qkv,
            [
                hf_config.hidden_size // hf_config.num_attention_heads,
                int(
                    (hf_config.num_key_value_heads / hf_config.num_attention_heads)
                    * hf_config.hidden_size
                    // hf_config.num_attention_heads
                ),
                int(
                    (hf_config.num_key_value_heads / hf_config.num_attention_heads)
                    * hf_config.hidden_size
                    // hf_config.num_attention_heads
                ),
            ],
            dim=2,
        )
        # splits along the (dims_per_head * (1 + 2 * kv-to-q head ratio)_ dim to get 3 tensors:
        # 1 x [TP_SIZE, NUM_Q_HEADS_PER_SHARD, dims_per_head, hidden_size] and 2 x [TP_SIZE, NUM_Q_HEADS_PER_SHARD, (dims_per_head / kv-to-q head ratio), hidden_size]
        # these are the Q, and K, V tensors respectively.

        # we have to do additional reshape for each individual tensor now,
        # into the expected square (or smaller than square, for K/V tensors) shape
        q, k, v = q.squeeze(dim=2), k.squeeze(dim=2), v.squeeze(dim=2)
        q = q.view(
            hf_config.num_attention_heads,
            hf_config.hidden_size // hf_config.num_attention_heads,
            hf_config.hidden_size,
        ).reshape(hf_config.hidden_size, hf_config.hidden_size)
        k = k.reshape(
            hf_config.num_key_value_heads,
            hf_config.hidden_size // hf_config.num_attention_heads,
            hf_config.hidden_size,
        ).reshape(
            hf_config.hidden_size
            // hf_config.num_attention_heads
            * hf_config.num_key_value_heads,
            hf_config.hidden_size,
        )
        v = v.reshape(
            hf_config.num_key_value_heads,
            hf_config.hidden_size // hf_config.num_attention_heads,
            hf_config.hidden_size,
        ).reshape(
            hf_config.hidden_size
            // hf_config.num_attention_heads
            * hf_config.num_key_value_heads,
            hf_config.hidden_size,
        )
        # print(f">> and finally get q:{q.size()}; k:{k.size()};v:{v.size()}") # BD
# 20240621 end
        # return these
        state_dict = {}
        for hf_key, proj in zip(hf_keys, [q, k, v]):
            state_dict[hf_key] = proj.clone()
        return state_dict


def convert(
    input_checkpoint_path,
    loaded_config,
    output_checkpoint_path,
    sequential: bool = True,
    precision: Literal["auto", "fp16", "bf16", "fp32"] = "auto",
    architecture: Literal["neox", "llama", "mistral"] = "neox",
):
    """convert a NeoX checkpoint to a HF model format.
    should perform model-parallel merging correctly
    but only supports features allowed by HF GPT-NeoX implementation (e.g. rotary embeddings)
    """

    ARCH = MODEL_KEYS[architecture]

    hf_config = create_config(loaded_config, architecture=architecture)

    hf_model = AutoModelForCausalLM.from_config(hf_config)

    if architecture == "neox":
        hf_transformer = hf_model.gpt_neox
    else:
        hf_transformer = hf_model.model

    # print(f">> hf model: {hf_model}")
    # print(f">> hf model.model: {hf_model.model}")

    if precision == "auto":
        print("Auto-detecting precision to save model into...")
        # save model in FP16 if Deepspeed fp16 was used in config, else 32 bit
        fp16 = get_key(loaded_config, "fp16")
        # fp16 = True # BD: temporary

        if fp16:
            try:
                # current behavior is to pass "fp16": {"enabled": true}, when using upstream Deepspeed
                if fp16["enabled"]:
                    hf_model.half()
                    print("Saving weights in fp16 precision...")
            except:
                try:
                    # attempt to access bf16 dict in yaml file, if fp16 not enabled
                    bf16 = get_key(loaded_config, "bf16")
                    # bf16 = get_key(loaded_config, "use_bfloat16")
                    if bf16:
                        hf_model.to(dtype=torch.bfloat16)
                        print("Saving weights in bf16 precision...")
                except:
                    hf_model.to(dtype=torch.float)
                    print(
                        "Model not trained in fp16 / bf16 mixed precision, saving weights in fp32..."
                    )
    else:
        name_to_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float,
        }
        print(f"Saving model into specified {precision} precision...")
        hf_model.to(dtype=name_to_dtype[precision])

    mp_partitions = get_key(loaded_config, "model-parallel-size")
    # mp_partitions = 2 # BD: temporary

    # Sequential saves all model states from an MP rank in one file.
    # so we only load the MP ranks only once and index into them with get_state().
    # for the pipeline-parallel case (pipeline-parallel-size >= 1),
    # we must load the correct layer's states at each step.
    # (this does mean that less memory is required for PP conversion.)
    loaded_tp_ranks = load_partitions(
        input_checkpoint_path, mp_partitions, layer_idx=1, sequential=sequential
    )
    ### Embedding layer ###
    # Embedding is layer idx 0
    if architecture == "neox":
        embed_in = hf_transformer.embed_in
    else:
        embed_in = hf_transformer.embed_tokens
    embed_in.load_state_dict(  # TODO: embed_in is not always model's name for embedding
        {
            "weight": torch.cat(
                get_state(
                    loaded_tp_ranks,
                    "word_embeddings.weight",
                    layer_idx=0,
                    sequential=sequential,
                ),
                dim=0,
            )
        }
    )
    assert (
        hf_config.vocab_size == embed_in.weight.shape[0]
    ), f"ERROR: calculated vocab size {hf_config.vocab_size} != embed param size {embed_in.shape[0]}"
    # print(f">> vocab_size and embedding shape 0: {hf_config.vocab_size}")
    ### End Embedding Layer ###

    for layer_i in tqdm(range(get_key(loaded_config, "num-layers"))):
    # for layer_i in tqdm(range(get_key(loaded_config, "n_layer"))):


        # get layer from hf model
        hf_layer = hf_transformer.layers[layer_i]  # TODO: model module names

        if not sequential:
            # in the non-sequential case, must load from each layer individually.
            # use layer index + 2 bc of embed layer and a dummy _pre_transformer_block, which are "layers 0 and 1"
            # BD: there isn't dummy_pre_transformer_block, yet starts with layer1, so still index + 2
            loaded_tp_ranks = load_partitions(
                input_checkpoint_path,
                mp_partitions,
                layer_idx=layer_i + 2,
                sequential=sequential,
            )

        # + 2 bc of embed layer and a dummy _pre_transformer_block
        state_dict = {}
        for key, hf_key in ARCH["ROW_PARALLEL_LINEAR_KEYS"].items():
            state_dict[hf_key] = torch.cat(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                ),
                dim=1,
            )
            # print(f"{hf_key}: {state_dict[hf_key].size()}") # BD

        # average layernorm stats over mp ranks
        for key, hf_key in ARCH["NORM_KEYS"].items():
            state_dict[hf_key] = sum(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
            ) / len(loaded_tp_ranks)

        # LinearWithTPMerge
        # BD: 这里的 h-4h 实际上是两层 
        # 推理的时候改成自定义的模型，不要强行转换
        # MLP 用 llama 的

        # 最快的方案 参考 llama3 的 modeling.py 把 llamaMLP 换成 GPT 的
        for key, hf_key in ARCH["COLUMN_PARALLEL_LINEAR_KEYS"].items():
            if "dense_h_to_4h" in key:
                param_per_tp = get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
                gate = torch.empty(0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                up = torch.empty(0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                for t in param_per_tp:
                    gate_p, up_p = t.chunk(2)
                    gate = torch.cat([gate, gate_p])
                    up = torch.cat([up, up_p])
                state_dict["mlp.gate_proj.weight"] = gate.to(torch.bfloat16).clone().detach().contiguous()
                state_dict["mlp.up_proj.weight"] = up.to(torch.bfloat16).clone().detach().contiguous()
                # print("mlp.gate_proj.weight: ", state_dict["mlp.gate_proj.weight"].size()) # BD
                # print("mlp.up_proj.weight: ",state_dict["mlp.up_proj.weight"].size()) # BD
            else:
                state_dict[hf_key] = torch.cat(
                    get_state(
                        loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                    ),
                    dim=0,
                )

        # LinearWithTPSplitBias
        for key, hf_key in ARCH["ROW_PARALLEL_BIAS_KEYS"].items():
            state_dict[hf_key] = sum(
                get_state(
                    loaded_tp_ranks, key, layer_idx=layer_i + 2, sequential=sequential
                )
            )

        # Just take one
        if "attention.bias" in hf_layer.state_dict():
            state_dict["attention.bias"] = hf_layer.state_dict()["attention.bias"]
        if "attention.masked_bias" in hf_layer.state_dict():
            state_dict["attention.masked_bias"] = hf_layer.state_dict()[
                "attention.masked_bias"
            ]

        # some architectures, like Mistral and Llama, have the following which must be handled specially:
        # - Q, K, V projections are performed separately, so we must split apart GPT-NeoX library's single QKV proj
        # - Support for Grouped-Query Attention, meaning the Q and the K, V projections may not be the same size
        if "GQA_QKV_KEYS" in ARCH:
            state_dict.update(
                reshard_and_split_qkv(
                    param_mapping=ARCH["GQA_QKV_KEYS"],
                    hf_config=hf_config,
                    loaded_tp_ranks=loaded_tp_ranks,
                    layer_idx=layer_i + 2,
                    sequential=sequential,
                )
            )
        # load state_dict into layer
        hf_layer.load_state_dict(state_dict)

    if not sequential:
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path,
            mp_partitions,
            get_key(loaded_config, "num-layers") + 2, # BD: final norm 
            # get_key(loaded_config, "n_layer") + 3,
            sequential=sequential,
        )
    # print(f">> loaded_layer26_ranks: {loaded_tp_ranks}") # BD

    # Load final layer norm
    # BD: num-layers+2对应最后的 final norm
    # BD: num-layers+3对应的是 lm head（output embedding）
    if architecture == "neox":
        lm_head = hf_model.embed_out
    else:
        lm_head = hf_model.lm_head
    norm_state_dict = {}
    for key, hf_key in ARCH["FINAL_NORM_KEYS"].items():
        norm_state_dict[hf_key] = sum(
            get_state(
                loaded_tp_ranks,
                key,
                layer_idx=get_key(loaded_config, "num-layers") + 3,
                # layer_idx=get_key(loaded_config, "n_layer") + 3,
                sequential=sequential,
            )
        ) / len(loaded_tp_ranks)

    if architecture == "neox":
        final_layer_norm = hf_transformer.final_layer_norm
    else:
        final_layer_norm = hf_transformer.norm
    # print(">> norm state dict:", norm_state_dict["weight"].size()) # BD
    final_layer_norm.load_state_dict(norm_state_dict)

    # Load output embedding
    if not sequential:
        loaded_tp_ranks = load_partitions(
            input_checkpoint_path,
            mp_partitions,
            get_key(loaded_config, "num-layers") + 3, # BD: lm head weight
            # get_key(loaded_config, "n_layer") + 4,
            sequential=sequential,
        )
    # output embedding / LM head
    if architecture == "neox":  # name of lm head / final linear proj varies
        lm_head = hf_model.embed_out
    else:
        lm_head = hf_model.lm_head
    lm_head.load_state_dict(
        {
            "weight": torch.cat(
                get_state(
                    loaded_tp_ranks,
                    "lm_head.weight",
                    layer_idx=get_key(loaded_config, "num-layers") + 4,
                    # layer_idx=get_key(loaded_config, "n_layer") + 4,
                    sequential=sequential,
                ),
                dim=0,
            ),
        }
    )

    del loaded_tp_ranks

    return hf_model


def main(input_args=None, overwrite_values=None):
    from huggingface_hub import create_repo, HfApi

    parser = argparse.ArgumentParser(
        description="Merge MP partitions and convert to HF Model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to NeoX checkpoint, e.g. /path/to/model/global_step143000",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file for the input NeoX checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output dir, where to save the HF Model, tokenizer, and configs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        help="What precision to save the model into. Defaults to auto, which auto-detects which 16-bit dtype to save into, or falls back to fp32.",
    )
    parser.add_argument(
        "--no_save_tokenizer",
        action="store_true",
        help="Whether to skip saving the tokenizer alongside a model.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="neox",
        help="What HF model class type to export into.",
    )
    args = parser.parse_args(input_args)

    # validate arguments
    assert args.precision in [
        "auto",
        "fp16",
        "bf16",
        "fp32",
    ], f"expected --precision to be one of 'auto', 'fp16', 'bf16', 'fp32' but got '{args.precision}' !"
    assert args.architecture in [
        "neox",
        "llama",
        "mistral",
    ], f"expected --architecture to be one of 'neox', 'mistral', 'llama', but got '{args.architecture}' !"

    if args.architecture == "llama":
        with open(args.config_file) as f:
            loaded_config = json.load(f)
            if overwrite_values:
                loaded_config.update(overwrite_values)
    else:
        with open(args.config_file) as f:
            raise KeyError("BD yaml")
            # loaded_config = yaml.full_load(f)
            if overwrite_values:
                loaded_config.update(overwrite_values)

    # Determine the checkpoint format of the model.
    # DeepSpeed saves models wrapped in a PipelineModule differently from those not.
    # PipelineModule models are saved as per-layer state dicts per TP shard,
    # while Sequential model state dicts are saved all together in one mp_rank_xx_model_states.pt
    # file per tensor/model parallel shard.
    pipeline_world_size = get_key(loaded_config, "pipe-parallel-size", 1)
    # pipeline_world_size = 2 # BD: temporary
    if pipeline_world_size == 0:
        sequential = True
        print(
            f"Detected 'pipe-parallel-size' of {pipeline_world_size}, assuming model is saved as Sequential..."
        )
    else:
        sequential = False
        print(
            f"Detected 'pipe-parallel-size' of {pipeline_world_size}, assuming model is saved as PipelineModule..."
        )

    # convert the model to HF.
    hf_model = convert(
        args.input_dir,
        loaded_config,
        args.output_dir,
        sequential=sequential,
        architecture=args.architecture,
    )

    # Save to disk.
    hf_model.save_pretrained(args.output_dir)

    if not args.no_save_tokenizer:
        # save tokenizer to directory as well, for easy loading of model as a HF model.
        # tokenizer_type = get_key(loaded_config, "tokenizer-type")
        tokenizer_type = "HFTokenizer" # BD: temporary

        if tokenizer_type == "HFTokenizer":  # TODO: handle sentencepiece tokenizers?
            print(f"saving tokenizer from file {get_key(loaded_config, 'vocab-file')}")
            print(
                "Warning: please check that your model config and tokenizer end with the correct special tokens (EOS, BOS)."
            )
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=get_key(loaded_config, "vocab-file")
            )
            print("loaded tokenizer: ", tokenizer)
            tokenizer.save_pretrained(args.output_dir)
            print("tokenizer saved!")


if __name__ == "__main__":

    # before running script:
    # `pip install --upgrade transformers`
    # `huggingface-cli login`
    #
    main()