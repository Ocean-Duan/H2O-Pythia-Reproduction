import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb
import math

# ================= 配置区域 =================
# 建议先用小一点的数据集跑通流程，比如 wikitext-2-raw-v1
DATASET_NAME = "wikitext" 
DATASET_CONFIG = "wikitext-2-raw-v1"
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_KV_COMPRESSION = True
# ===========================================

# ================= H2O 算法配置 =================
# 建议配置：Window=256, Heavy=256 -> 总缓存 512
# 相比 Baseline (1024或更多) 能看到明显的显存停止增长
H2O_RECENT_SIZE = 256
H2O_HEAVY_SIZE = 256
# ===============================================

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
    # 1. 参数获取
    num_attention_heads = self.config.num_attention_heads
    hidden_size = self.config.hidden_size
    head_size = self.head_size 

    bsz, q_len, _ = hidden_states.size()

    # 2. 投影 & Reshape
    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(bsz, q_len, num_attention_heads, 3 * head_size)
    qkv = qkv.permute(0, 2, 1, 3) 
    query, key, value = qkv.chunk(3, dim=-1)

    # 3. RoPE
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value, seq_len=key.shape[-2])
    else:
        cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # 4. Update Cache
    if layer_past is not None:
        # 这一步会自动更新 layer_past.layers[self.layer_idx] 里的 keys/values
        key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs={"cache_position": cache_position})

    # ================= H2O 核心逻辑 =================
    seq_len = key.shape[2]
    current_limit = H2O_RECENT_SIZE + H2O_HEAVY_SIZE
    
    # 仅在生成阶段 (q_len == 1) 且缓存溢出时压缩
    if q_len == 1 and seq_len > current_limit:
        
        # --- A. 计算分数 ---
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(head_size)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        # --- B. 维护历史分数 ---
        if not hasattr(layer_past, "h2o_scores"):
            layer_past.h2o_scores = torch.zeros((bsz, num_attention_heads, seq_len), dtype=torch.float32, device=key.device)
            layer_past.h2o_scores[:, :, :seq_len] += attn_weights.squeeze(-2)
        else:
            prev_len = layer_past.h2o_scores.shape[-1]
            if seq_len > prev_len:
                diff = seq_len - prev_len
                new_zeros = torch.zeros((bsz, num_attention_heads, diff), device=key.device, dtype=torch.float32)
                layer_past.h2o_scores = torch.cat([layer_past.h2o_scores, new_zeros], dim=-1)
            layer_past.h2o_scores += attn_weights.squeeze(-2)

        # --- C. 驱逐策略 (Eviction) ---
        total_scores = layer_past.h2o_scores.sum(dim=1)
        temp_scores = total_scores.clone()
        temp_scores[:, -H2O_RECENT_SIZE:] = -float('inf')
        
        _, heavy_indices = torch.topk(temp_scores, k=H2O_HEAVY_SIZE, dim=-1)
        recent_indices = torch.arange(seq_len - H2O_RECENT_SIZE, seq_len, device=key.device).expand(bsz, -1)
        
        kept_indices = torch.cat([heavy_indices, recent_indices], dim=-1)
        kept_indices, _ = kept_indices.sort(dim=-1)

        # --- D. Pruning ---
        def gather_kv(tensor, idx):
            idx_expanded = idx.unsqueeze(1).unsqueeze(-1).expand(-1, tensor.size(1), -1, tensor.size(3))
            return torch.gather(tensor, 2, idx_expanded)
        
        pruned_key = gather_kv(key, kept_indices)
        pruned_value = gather_kv(value, kept_indices)
        
        idx_scores = kept_indices.unsqueeze(1).expand(-1, num_attention_heads, -1)
        pruned_scores = torch.gather(layer_past.h2o_scores, 2, idx_scores)
        
        # === 终极修复点：修改对象属性而非替换对象 ===
        # 获取当前层的 Layer 对象
        layer_obj = layer_past.layers[self.layer_idx]
        # 修改其内部存储的 keys 和 values
        layer_obj.keys = pruned_key
        layer_obj.values = pruned_value
        
        # 更新分数和局部变量
        layer_past.h2o_scores = pruned_scores
        key = pruned_key
        value = pruned_value
        
    # ================= H2O 结束 =================

    # 5. 最终计算 Attention (含 PPL 修复)
    is_causal_masking = (q_len > 1)
    attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal_masking 
    )

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    attn_output = attn_output.view(bsz, q_len, hidden_size)
    attn_output = self.dense(attn_output)

    return (attn_output, (key, value) if output_attentions else None)

def enable_h2o_monkey_patch(model):
    print(f"\n>>> 正在应用 H2O Monkey Patch (修复版)...")
    print(f">>> 配置: Window={H2O_RECENT_SIZE}, Heavy={H2O_HEAVY_SIZE}, Limit={H2O_RECENT_SIZE + H2O_HEAVY_SIZE}")
    
    # 遍历替换
    count = 0
    for layer in model.gpt_neox.layers:
        import types
        layer.attention.forward = types.MethodType(h2o_gpt_neox_attention_forward, layer.attention)
        count += 1
    
    print(f">>> 成功替换了 {count} 层 Attention 的 forward 方法。\n")

def print_memory_usage(stage_name=""):
    """打印当前显存占用和峰值显存占用"""
    if not torch.cuda.is_available():
        print("CUDA不可用，无法监测显存")
        return
    
    # 强制同步，确保测量准确
    torch.cuda.synchronize()
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"[{stage_name}] 当前显存: {allocated:.2f} GB | 峰值显存: {peak:.2f} GB")

def calculate_ppl(model, tokenizer, text_list, max_length=2048, stride=512):
    """
    计算困惑度 (Perplexity)
    使用滑动窗口策略处理长文本
    """
    encodings = tokenizer("\n\n".join(text_list), return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    
    nlls = [] # Negative Log Likelihoods
    
    print(f"开始计算 PPL，总 Token 数: {input_ids.size(1)}...")
    
    # 进度条
    for i in tqdm(range(0, input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i    # target_len 是我们需要预测的长度
        
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        
        # 将 target 中不属于我们要预测的部分设为 -100 (PyTorch 会自动忽略这些位置的 Loss)
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            # CrossEntropyLoss 默认是平均值，我们需要乘回去得到总 Loss
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        if end_loc == input_ids.size(1):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# ================= 主程序 =================
if __name__ == "__main__":
    


    print("-" * 40)
    print(f"检测到设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print("-" * 40)

    # 1. 加载模型
    print(f"正在加载模型: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype=torch.float16, 
        device_map="auto", 
        cache_dir=CACHE_DIR
    )

    # 初始显存（模型权重占用）
    print_memory_usage("模型加载后")

    # 2. 准备 PPL 测试数据 (加载 wikitext 的 test 集的前 1% 用于快速测试)
    print("正在加载数据集...")
    data = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", cache_dir=CACHE_DIR)
    # 为了演示速度，这里只取前几个长文本，实际实验请跑全量或更多数据
    test_texts = [x['text'] for x in data if len(x['text']) > 100][:5] 

    enable_h2o_monkey_patch(model)

    # 3. 计算 PPL
    print_memory_usage("PPL计算前")
    ppl = calculate_ppl(model, tokenizer, test_texts)
    print(f"\n>>> Baseline PPL: {ppl:.2f}")

    # 4. 显存测试：长文本生成 (模拟 KV Cache 增长)
    print("\n开始长文本生成测试 (测试 KV Cache 显存压力)...")
    # 重置峰值统计，只统计生成过程中的显存
    torch.cuda.reset_peak_memory_stats()

    input_text = "The history of natural language processing dates back to"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=2048,    # 生成越长，KV Cache 越大，H2O 的效果越明显
            do_sample=True, 
            use_cache=True         # 必须开启 KV Cache
        )

    print_memory_usage("生成结束后")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成文本预览: {generated_text[:100]}...")