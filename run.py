import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb
import math
import time
import pandas as pd
import gc

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
    # 1. 参数获取 & 基础设置
    num_attention_heads = self.config.num_attention_heads
    hidden_size = self.config.hidden_size
    head_size = self.head_size 

    bsz, q_len, _ = hidden_states.size()

    # 2. 投影 & Reshape (QKV 计算)
    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(bsz, q_len, num_attention_heads, 3 * head_size)
    qkv = qkv.permute(0, 2, 1, 3) 
    query, key, value = qkv.chunk(3, dim=-1)

    # 3. RoPE 位置编码
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value, seq_len=key.shape[-2])
    else:
        cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # 4. Update Cache (将当前 token 加入 KV Cache)
    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs={"cache_position": cache_position})

    # ================= KV Cache 压缩核心逻辑 (兼容 Baseline/Local/H2O) =================
    seq_len = key.shape[2]
    # 当前允许的最大缓存长度
    current_limit = H2O_RECENT_SIZE + H2O_HEAVY_SIZE
    
    # 触发条件：
    # 1. 开启了压缩开关 (ENABLE_KV_COMPRESSION)
    # 2. 处于生成阶段 (q_len == 1)
    # 3. 当前序列长度超过了设定的阈值
    if ENABLE_KV_COMPRESSION and q_len == 1 and seq_len > current_limit:
        
        kept_indices = None
        
        # --- 分支 A: Local Attention (纯滑动窗口) ---
        # 如果 Heavy Size 设为 0，我们不需要计算注意力分数，直接切片即可 (性能优化)
        if H2O_HEAVY_SIZE == 0:
            # 直接保留最近的 H2O_RECENT_SIZE 个 Token
            kept_indices = torch.arange(seq_len - H2O_RECENT_SIZE, seq_len, device=key.device).expand(bsz, -1)
            
            # 如果是纯 Local 模式，不需要维护 h2o_scores
            if hasattr(layer_past, "h2o_scores"): 
                del layer_past.h2o_scores

        # --- 分支 B: H2O (Heavy Hitters + Recent) ---
        else:
            # 1. 计算 Attention Score (仅为了挑选 Heavy Hitters)
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            attn_weights = attn_weights / math.sqrt(head_size)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

            # 2. 维护历史分数累加
            if not hasattr(layer_past, "h2o_scores"):
                layer_past.h2o_scores = torch.zeros((bsz, num_attention_heads, seq_len), dtype=torch.float32, device=key.device)
                layer_past.h2o_scores[:, :, :seq_len] += attn_weights.squeeze(-2)
            else:
                prev_len = layer_past.h2o_scores.shape[-1]
                if seq_len > prev_len: # 显存未压缩前的短暂增长需要补齐 score
                    diff = seq_len - prev_len
                    new_zeros = torch.zeros((bsz, num_attention_heads, diff), device=key.device, dtype=torch.float32)
                    layer_past.h2o_scores = torch.cat([layer_past.h2o_scores, new_zeros], dim=-1)
                layer_past.h2o_scores += attn_weights.squeeze(-2)

            # 3. 驱逐策略 (Eviction)
            total_scores = layer_past.h2o_scores.sum(dim=1) # 对 Head 维度求和
            temp_scores = total_scores.clone()
            
            # 保护 Recent Token 不参与 Heavy 竞争 (先设为无穷小)
            temp_scores[:, -H2O_RECENT_SIZE:] = -float('inf')
            
            # 选出 Heavy Hitters
            _, heavy_indices = torch.topk(temp_scores, k=H2O_HEAVY_SIZE, dim=-1)
            
            # 选出 Recent Tokens
            recent_indices = torch.arange(seq_len - H2O_RECENT_SIZE, seq_len, device=key.device).expand(bsz, -1)
            
            # 合并索引并排序
            kept_indices = torch.cat([heavy_indices, recent_indices], dim=-1)
            kept_indices, _ = kept_indices.sort(dim=-1)

        # --- 执行压缩 (Pruning) ---
        # 无论是 Local 还是 H2O，最后都通过 kept_indices 进行 gather 操作
        if kept_indices is not None:
            def gather_kv(tensor, idx):
                idx_expanded = idx.unsqueeze(1).unsqueeze(-1).expand(-1, tensor.size(1), -1, tensor.size(3))
                return torch.gather(tensor, 2, idx_expanded)
            
            pruned_key = gather_kv(key, kept_indices)
            pruned_value = gather_kv(value, kept_indices)
            
            # 如果是 H2O 模式，还需要压缩分数缓存
            if H2O_HEAVY_SIZE > 0 and hasattr(layer_past, "h2o_scores"):
                idx_scores = kept_indices.unsqueeze(1).expand(-1, num_attention_heads, -1)
                pruned_scores = torch.gather(layer_past.h2o_scores, 2, idx_scores)
                layer_past.h2o_scores = pruned_scores
            
            # 修改 KV Cache 对象
            layer_obj = layer_past.layers[self.layer_idx]
            layer_obj.keys = pruned_key
            layer_obj.values = pruned_value
            
            # 更新当前函数的局部变量，用于后续 Attention 计算
            key = pruned_key
            value = pruned_value

    # ================= 压缩逻辑结束 =================

    # 5. 最终计算 Attention (标准计算流程)
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

def run_generation_benchmark(model, tokenizer, prompt_text, new_tokens=512):
    """
    运行生成测试，测量显存占用和推理速度 (TPOT)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # 强制回收显存，确保测量准确
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    with torch.no_grad():
        # 注意：这里 min_new_tokens 设置为 new_tokens 确保生成长度一致
        outputs = model.generate(
            **inputs, 
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            do_sample=False, # 测速建议用贪婪搜索，减少随机性
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    
    # 统计指标
    total_time = end_time - start_time
    # 实际生成的 token 数量 (outputs 包含 input，所以要减去)
    generated_count = outputs.shape[1] - inputs.input_ids.shape[1]
    tpot = (total_time / generated_count) * 1000 # 毫秒/token
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3 # GB
    
    return peak_memory, tpot

# ================= 主程序 =================
if __name__ == "__main__":
    print("-" * 40)
    print(f"检测到设备: {DEVICE}")
    print("-" * 40)

    # 1. 加载模型 (只加载一次)
    print(f"正在加载模型: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    # 确保 pad_token 存在，否则生成会报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        dtype=torch.float16, 
        device_map="auto", 
        cache_dir=CACHE_DIR
    )
    
    # 应用 Monkey Patch (只需要应用一次，后续通过全局变量控制行为)
    enable_h2o_monkey_patch(model)

    # 2. 准备测试数据
    print("正在加载数据集 (Wikitext test set)...")
    data = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", cache_dir=CACHE_DIR)
    # 选取部分长文本用于 PPL 测试 (为了速度，这里取前 3 个长文本，正式实验建议取更多)
    ppl_test_texts = [x['text'] for x in data if len(x['text']) > 100][:3]
    
    # 准备生成测试的 Prompt (模拟长 Context)
    gen_prompt = "The history of artificial intelligence began with myths, stories and rumors of master craftsmen endowed with intelligence by the gods."

    # 3. 定义实验配置列表
    # 这里的 total_cache = recent + heavy
    # Baseline: 全量缓存 (ENABLE_KV_COMPRESSION = False)
    # H2O: Recent + Heavy
    # Local: 仅 Recent (Heavy = 0)
    
    experiments = [
        {
            "name": "Full Cache (Baseline)",
            "enable_compression": False,
            "recent": 0, "heavy": 0 
        },
        {
            "name": "H2O (Budget=512)", 
            "enable_compression": True,
            "recent": 256, "heavy": 256
        },
        {
            "name": "H2O (Budget=256)", 
            "enable_compression": True,
            "recent": 128, "heavy": 128
        },
        {
            "name": "Local Attention (Budget=512)", 
            "enable_compression": True,
            "recent": 512, "heavy": 0 # Heavy=0 退化为滑动窗口
        },
        {
            "name": "Local Attention (Budget=256)", 
            "enable_compression": True,
            "recent": 256, "heavy": 0
        },
    ]
    
    results = []

    print("\n>>> 开始自动化评测 Pipeline...\n")
    
    for exp in experiments:
        exp_name = exp["name"]
        print(f"正在运行实验: [{exp_name}]")
        
        # --- A. 修改全局配置变量 ---
        # 这里的修改会直接影响到 h2o_gpt_neox_attention_forward 函数
        ENABLE_KV_COMPRESSION = exp["enable_compression"]
        H2O_RECENT_SIZE = exp["recent"]
        H2O_HEAVY_SIZE = exp["heavy"]
        
        print(f"   配置: Compress={ENABLE_KV_COMPRESSION}, Recent={H2O_RECENT_SIZE}, Heavy={H2O_HEAVY_SIZE}")

        # --- B. 运行 PPL 测试 ---
        # 清理缓存，防止上一个实验残留
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            ppl = calculate_ppl(model, tokenizer, ppl_test_texts, stride=512)
            print(f"   -> PPL: {ppl:.2f}")
        except Exception as e:
            print(f"   -> PPL 计算出错: {e}")
            ppl = float("nan")

        # --- C. 运行生成测试 (显存 & 速度) ---
        torch.cuda.empty_cache()
        try:
            # 生成 512 个 token 来观察稳定状态的显存
            peak_mem, tpot = run_generation_benchmark(model, tokenizer, gen_prompt, new_tokens=512)
            print(f"   -> Peak Memory: {peak_mem:.2f} GB")
            print(f"   -> TPOT: {tpot:.2f} ms/token")
        except Exception as e:
            print(f"   -> 生成测试出错: {e}")
            peak_mem = float("nan")
            tpot = float("nan")
            
        # 记录结果
        results.append({
            "Experiment": exp_name,
            "PPL": ppl,
            "Peak Memory (GB)": peak_mem,
            "TPOT (ms)": tpot,
            "Recent": H2O_RECENT_SIZE,
            "Heavy": H2O_HEAVY_SIZE
        })
        print("-" * 30)

    # 4. 输出汇总表格并保存
    df = pd.DataFrame(results)
    print("\n================ 最终实验结果 ================")
    print(df.to_markdown(index=False))
    
    df.to_csv("experiment_results.csv", index=False)
    print("\n结果已保存至 experiment_results.csv")