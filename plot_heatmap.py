import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置 =================
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# [核心修改] 要遍历的层数列表
LAYERS_TO_PLOT = [2, 16, 30] 

# 更加复杂的文本 (Wikitext 片段)
COMPLEX_TEXT = """The game of billiards has a rich history starting from the 15th century in Northern Europe as an outdoor lawn game similar to croquet. The game moved indoors to a wooden table with green cloth to simulate grass, and a simple border was placed around the edges. The balls were shoved, rather than struck, with wooden sticks called "maces"."""

CAPTURED_ATTN = {} # 字典存储: {layer_idx: matrix}
# =======================================

def h2o_capture_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    layer_past=None,
    output_attentions: bool = False,
    position_embeddings=None,
    **kwargs
):
    bsz, q_len, _ = hidden_states.size()
    num_attention_heads = self.config.num_attention_heads
    head_size = self.head_size 

    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(bsz, q_len, num_attention_heads, 3 * head_size)
    qkv = qkv.permute(0, 2, 1, 3) 
    query, key, value = qkv.chunk(3, dim=-1)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value, seq_len=key.shape[-2])
    else:
        cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx)

    # 捕获逻辑：如果在目标列表中，且处于 decoding 阶段 (q_len=1)
    if self.layer_idx in LAYERS_TO_PLOT and q_len == 1:
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_size)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        # 存入字典
        global CAPTURED_ATTN
        CAPTURED_ATTN[self.layer_idx] = attn_weights[0, :, 0, :].detach().cpu().numpy()

    # 继续前向传播
    attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, -1)
    attn_output = self.dense(attn_output)
    return (attn_output, (key, value) if output_attentions else None)

def inject_monitors(model):
    print(f">>> Hooking layers: {LAYERS_TO_PLOT}...")
    for layer in model.gpt_neox.layers:
        import types
        layer.attention.forward = types.MethodType(h2o_capture_forward, layer.attention)

def plot_heatmap(layer_idx, tokens, attn_matrix):
    plt.figure(figsize=(24, 8)) # 宽一点适合长句
    
    # 截取前 100 个 token 以免太挤
    limit = min(100, len(tokens))
    display_tokens = [t.replace('Ġ', '').strip()[:8] for t in tokens[:limit]]
    matrix_subset = attn_matrix[:, :limit]

    # 动态调整 vmax：取 95 分位数，避免某个极亮值导致全图变黑
    import numpy as np
    dynamic_vmax = np.percentile(matrix_subset, 98) 
    
    sns.heatmap(matrix_subset, cmap="viridis", vmin=0, vmax=dynamic_vmax,
                xticklabels=display_tokens, yticklabels=False,
                cbar_kws={"label": "Score"})

    plt.title(f"Layer {layer_idx} Attention Map", fontsize=18)
    plt.xlabel("Context Tokens", fontsize=14)
    plt.ylabel("Heads", fontsize=14)
    plt.xticks(rotation=90, fontsize=9)
    
    save_path = f"heatmap_layer_{layer_idx}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f">>> Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=CACHE_DIR)
    inject_monitors(model)
    
    inputs = tokenizer(COMPLEX_TEXT, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    print(">>> Running inference (Generating 2 tokens to trigger decode step)...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    for layer_idx in LAYERS_TO_PLOT:
        if layer_idx in CAPTURED_ATTN:
            plot_heatmap(layer_idx, tokens, CAPTURED_ATTN[layer_idx])
        else:
            print(f"Warning: Layer {layer_idx} not captured.")