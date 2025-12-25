import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
import math
import time
import pandas as pd
import gc

# ================= 配置区域 =================
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 全局变量
ENABLE_KV_COMPRESSION = False
H2O_RECENT_SIZE = 64
H2O_HEAVY_SIZE = 64
LAYER_KV_SIZES = {}
# ===========================================

# ... (此处省略 h2o_gpt_neox_attention_forward 函数，与之前完全一致，无需改动) ...
# 为了完整性，请确保这里包含之前修复好的 h2o_gpt_neox_attention_forward 函数
# 以及 enable_h2o_monkey_patch 函数

def h2o_gpt_neox_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: torch.FloatTensor = None,
    layer_past=None,
    output_attentions: bool = False,
    cache_position: torch.LongTensor = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    **kwargs
):
    bsz, q_len, _ = hidden_states.size()
    num_attention_heads = self.config.num_attention_heads
    hidden_size = self.config.hidden_size
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
        key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs={"cache_position": cache_position})

    seq_len = key.shape[2]
    current_limit = H2O_RECENT_SIZE + H2O_HEAVY_SIZE
    
    if ENABLE_KV_COMPRESSION and q_len == 1 and seq_len > current_limit:
        kept_indices = None
        if H2O_HEAVY_SIZE == 0:
            kept_indices = torch.arange(seq_len - H2O_RECENT_SIZE, seq_len, device=key.device).expand(bsz, -1)
            if hasattr(layer_past, "h2o_scores"): del layer_past.h2o_scores
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_size)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            if not hasattr(layer_past, "h2o_scores"):
                layer_past.h2o_scores = torch.zeros((bsz, num_attention_heads, seq_len), dtype=torch.float32, device=key.device)
                layer_past.h2o_scores[:, :, :seq_len] += attn_weights.squeeze(-2)
            else:
                prev_len = layer_past.h2o_scores.shape[-1]
                if seq_len > prev_len:
                    diff = seq_len - prev_len
                    layer_past.h2o_scores = torch.cat([layer_past.h2o_scores, torch.zeros((bsz, num_attention_heads, diff), device=key.device)], dim=-1)
                layer_past.h2o_scores += attn_weights.squeeze(-2)

            total_scores = layer_past.h2o_scores.sum(dim=1).clone()
            total_scores[:, -H2O_RECENT_SIZE:] = -float('inf')
            _, heavy_indices = torch.topk(total_scores, k=H2O_HEAVY_SIZE, dim=-1)
            recent_indices = torch.arange(seq_len - H2O_RECENT_SIZE, seq_len, device=key.device).expand(bsz, -1)
            kept_indices = torch.cat([heavy_indices, recent_indices], dim=-1)
            kept_indices, _ = kept_indices.sort(dim=-1)

        if kept_indices is not None:
            def gather_kv(tensor, idx):
                idx_expanded = idx.unsqueeze(1).unsqueeze(-1).expand(-1, tensor.size(1), -1, tensor.size(3))
                return torch.gather(tensor, 2, idx_expanded)
            
            pruned_key = gather_kv(key, kept_indices)
            pruned_value = gather_kv(value, kept_indices)
            
            if H2O_HEAVY_SIZE > 0 and hasattr(layer_past, "h2o_scores"):
                idx_scores = kept_indices.unsqueeze(1).expand(-1, num_attention_heads, -1)
                layer_past.h2o_scores = torch.gather(layer_past.h2o_scores, 2, idx_scores)

            if isinstance(layer_past, (DynamicCache, Cache)):
                if hasattr(layer_past, "key_cache") and len(layer_past.key_cache) > self.layer_idx:
                    layer_past.key_cache[self.layer_idx] = pruned_key
                    layer_past.value_cache[self.layer_idx] = pruned_value
            
            key = pruned_key
            value = pruned_value

    kv_size_bytes = key.element_size() * key.nelement() + value.element_size() * value.nelement()
    global LAYER_KV_SIZES
    LAYER_KV_SIZES[self.layer_idx] = kv_size_bytes

    is_causal_masking = (q_len > 1)
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal_masking 
    )
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, hidden_size)
    attn_output = self.dense(attn_output)
    return (attn_output, (key, value) if output_attentions else None)

def enable_h2o_monkey_patch(model):
    print(f"\n>>> [System] Injecting H2O Logic...")
    for layer in model.gpt_neox.layers:
        import types
        layer.attention.forward = types.MethodType(h2o_gpt_neox_attention_forward, layer.attention)

def get_real_long_text():
    # 使用 Wikitext-2
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"][:15])
        return text
    except:
        return "The history of AI " * 500

def run_benchmark(model, tokenizer, text, exp_config):
    """
    统一运行：显存、TPOT、PPL
    """
    global ENABLE_KV_COMPRESSION, H2O_RECENT_SIZE, H2O_HEAVY_SIZE, LAYER_KV_SIZES
    
    # 设置参数
    ENABLE_KV_COMPRESSION = exp_config["compress"]
    H2O_RECENT_SIZE = exp_config["r"]
    H2O_HEAVY_SIZE = exp_config["h"]
    
    # ---------------- Step 1: 速度 & 显存测试 (长生成) ----------------
    print(f"   [Running] {exp_config['name']} - Speed & Memory...")
    # 截取 Prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if inputs.input_ids.shape[1] > 100:
        inputs.input_ids = inputs.input_ids[:, :100]
        inputs.attention_mask = inputs.attention_mask[:, :100]
        
    LAYER_KV_SIZES = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 计时开始
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            min_new_tokens=1024, # 缩短一点以加快实验速度，或者保持 2000
            max_new_tokens=1024, 
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算指标
    total_time = end_time - start_time
    tpot = (total_time / 1024) * 1000 # ms/token
    throughput = 1024 / total_time    # tokens/sec
    effective_kv_gb = sum(LAYER_KV_SIZES.values()) / 1024**3
    
    # ---------------- Step 2: PPL 测试 ----------------
    print(f"   [Running] {exp_config['name']} - PPL...")
    # 清理显存
    torch.cuda.empty_cache()
    
    # PPL 文本长度，确保超过 Cache 上限
    eval_len = 1500 
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)[:, :eval_len]
    
    nlls = []
    past_key_values = None
    prev_token = input_ids[:, 0:1]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    with torch.no_grad():
        for i in tqdm(range(1, input_ids.size(1)), leave=False):
            target_token = input_ids[:, i:i+1]
            outputs = model(prev_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            loss = loss_fct(logits, target_token.view(-1))
            nlls.append(loss)
            prev_token = target_token
            
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    return {
        "Experiment": exp_config["name"],
        "Budget (Tokens)": H2O_RECENT_SIZE + H2O_HEAVY_SIZE if ENABLE_KV_COMPRESSION else "Full",
        "KV Size (GB)": effective_kv_gb,
        "TPOT (ms)": tpot,
        "Throughput (T/s)": throughput,
        "PPL": ppl
    }

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto", cache_dir=CACHE_DIR)
    enable_h2o_monkey_patch(model)
    long_text = get_real_long_text()

    # ================= 定义所有实验组 =================
    all_experiments = []

    # 1. Baseline
    all_experiments.append({"name": "Baseline", "compress": False, "r": 0, "h": 0})

    # 2. 核心对比 (之前的实验)
    all_experiments.append({"name": "H2O (64+64)", "compress": True, "r": 64, "h": 64})
    all_experiments.append({"name": "Local (128)", "compress": True, "r": 128, "h": 0})

    # 3. Budget Sweep (用于画曲线图)
    all_experiments.append({"name": "H2O (128+128)", "compress": True, "r": 128, "h": 128})
    all_experiments.append({"name": "H2O (256+256)", "compress": True, "r": 256, "h": 256})

    # 4. Ratio Ablation (用于参数分析) - 固定 Total=256
    # 对比 H2O (128+128) vs Local (256) vs Heavy-Heavy (64+192)
    all_experiments.append({"name": "Local (256)", "compress": True, "r": 256, "h": 0})
    all_experiments.append({"name": "H2O (64r+192h)", "compress": True, "r": 64, "h": 192})

    results = []
    print("\n>>> Starting Full NeurIPS Experiments Suite...\n")
    
    for exp in all_experiments:
        res = run_benchmark(model, tokenizer, long_text, exp)
        results.append(res)
        print(f"   -> Done. PPL: {res['PPL']:.2f}, KV: {res['KV Size (GB)']:.4f} GB")

    df = pd.DataFrame(results)
    print("\n================ FINAL PAPER RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv("neurips_experiments.csv", index=False)