import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
import math
import types
import time
from src import h2o_state

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
    # 计时辅助函数
    #do_profile = (hidden_states.size(1) == 1)
    
    #if do_profile: torch.cuda.synchronize()
    #t_start = time.time()

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
    
    # 记录 Prep 时间
    #if do_profile:
    #    torch.cuda.synchronize()
    #    t_prep = time.time()

    seq_len = key.shape[2]
    current_limit = h2o_state.H2O_RECENT_SIZE + h2o_state.H2O_HEAVY_SIZE
    
    # 初始化分段计时变量，防止未进入分支导致报错
    #t_score = t_prep if do_profile else 0
    #t_evict = t_prep if do_profile else 0
    #t_gather = t_prep if do_profile else 0

    if h2o_state.ENABLE_KV_COMPRESSION and q_len == 1 and seq_len > current_limit:
        kept_indices = None
        if h2o_state.H2O_HEAVY_SIZE == 0:
            key = key[:, :, -h2o_state.H2O_RECENT_SIZE:].clone()
            value = value[:, :, -h2o_state.H2O_RECENT_SIZE:].clone()
            # Local 模式没有复杂逻辑，时间点直接顺延
            #if do_profile:
            #    t_score = t_evict = t_gather = time.time()
        else:
            # 1. 计算 Score
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
            
            # 记录 Score 时间
            #if do_profile:
            #    torch.cuda.synchronize()
            #    t_score = time.time()

            # 2. TopK Eviction
            total_scores = layer_past.h2o_scores.sum(dim=1).clone()
            total_scores[:, -h2o_state.H2O_RECENT_SIZE:] = -float('inf')
            _, heavy_indices = torch.topk(total_scores, k=h2o_state.H2O_HEAVY_SIZE, dim=-1)
            recent_indices = torch.arange(seq_len - h2o_state.H2O_RECENT_SIZE, seq_len, device=key.device).expand(bsz, -1)
            kept_indices = torch.cat([heavy_indices, recent_indices], dim=-1)
            kept_indices, _ = kept_indices.sort(dim=-1)
            
            # 记录 Eviction 时间
            #if do_profile:
            #    torch.cuda.synchronize()
            #    t_evict = time.time()

        if kept_indices is not None:
            def gather_kv(tensor, idx):
                idx_expanded = idx.unsqueeze(1).unsqueeze(-1).expand(-1, tensor.size(1), -1, tensor.size(3))
                return torch.gather(tensor, 2, idx_expanded)
            
            pruned_key = gather_kv(key, kept_indices)
            pruned_value = gather_kv(value, kept_indices)
            
            if h2o_state.H2O_HEAVY_SIZE > 0 and hasattr(layer_past, "h2o_scores"):
                idx_scores = kept_indices.unsqueeze(1).expand(-1, num_attention_heads, -1)
                layer_past.h2o_scores = torch.gather(layer_past.h2o_scores, 2, idx_scores)

            if isinstance(layer_past, (DynamicCache, Cache)):
                if hasattr(layer_past, "key_cache") and len(layer_past.key_cache) > self.layer_idx:
                    layer_past.key_cache[self.layer_idx] = pruned_key
                    layer_past.value_cache[self.layer_idx] = pruned_value
            
            key = pruned_key
            value = pruned_value
            
            # 记录 Gather 时间
            #if do_profile:
            #    torch.cuda.synchronize()
            #    t_gather = time.time()
    
    # 填补未进入分支时的时间空缺
    #if do_profile:
    #    if t_score == 0: t_score = t_prep
    #    if t_evict == 0: t_evict = t_score
    #    if t_gather == 0: t_gather = t_evict

    # 统计 KV Size
    kv_size_bytes = key.element_size() * key.nelement() + value.element_size() * value.nelement()
    h2o_state.LAYER_KV_SIZES[self.layer_idx] = kv_size_bytes

    is_causal_masking = (q_len > 1)
    attn_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=is_causal_masking 
    )
    
    # 记录 Attention Compute 时间
    #if do_profile:
    #    torch.cuda.synchronize()
    #    t_attn = time.time()
        
        # 累加到全局状态
    #    h2o_state.PROFILE_STATS["prep"] += (t_prep - t_start)
    #    h2o_state.PROFILE_STATS["score"] += (t_score - t_prep)
    #    h2o_state.PROFILE_STATS["evict"] += (t_evict - t_score)
    #    h2o_state.PROFILE_STATS["gather"] += (t_gather - t_evict)
    #    h2o_state.PROFILE_STATS["attn"] += (t_attn - t_gather)
    #    h2o_state.PROFILE_STATS["total"] += (t_attn - t_start)
    #    h2o_state.PROFILE_STEPS += 1

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, hidden_size)
    attn_output = self.dense(attn_output)
    return (attn_output, (key, value) if output_attentions else None)

def enable_h2o_monkey_patch(model):
    print(f"\n>>> [System] Injecting H2O Logic...")
    for layer in model.gpt_neox.layers:
        layer.attention.forward = types.MethodType(h2o_gpt_neox_attention_forward, layer.attention)