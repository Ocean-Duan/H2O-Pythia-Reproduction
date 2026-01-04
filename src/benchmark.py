import torch
import time
import gc
from tqdm import tqdm
from src import h2o_state, config

def run_benchmark(model, tokenizer, text, exp_config):
    # === 修改全局状态 ===
    h2o_state.ENABLE_KV_COMPRESSION = exp_config["compress"]
    h2o_state.H2O_RECENT_SIZE = exp_config["r"]
    h2o_state.H2O_HEAVY_SIZE = exp_config["h"]
    h2o_state.reset_stats() 
    
    prompt_len = config.BENCHMARK_CONFIG["speed_test_prompt_len"]
    gen_len = config.BENCHMARK_CONFIG["speed_test_gen_len"]
    ppl_len = config.BENCHMARK_CONFIG["ppl_eval_seq_len"]

    # ---------------- Speed and Mem Test ----------------
    print(f"   [Running] {exp_config['name']} - Speed & Memory...")
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    if inputs.input_ids.shape[1] > prompt_len:
        inputs.input_ids = inputs.input_ids[:, :prompt_len]
        inputs.attention_mask = inputs.attention_mask[:, :prompt_len]
    else:
        print(f"Warning: Input text is shorter ({inputs.input_ids.shape[1]}) than requested prompt_len ({prompt_len})")

    # Warm up
    print("   [Warmup] ...")
    model.generate(
        input_ids=inputs.input_ids[:, :10], 
        attention_mask=inputs.attention_mask[:, :10],
        max_new_tokens=10, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    cur_len = inputs.input_ids.shape[1]
    total_target = cur_len + gen_len
    
    if total_target > 2048:
        print(f"   [Error] Total tokens {total_target} exceeds model limit 2048! Adjust config.")
        return { "Experiment": exp_config["name"], "PPL": -1, "TPOT (ms)": -1 }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            min_new_tokens=gen_len,
            max_new_tokens=gen_len,
            do_sample=False, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    tpot = (total_time / gen_len) * 1000
    throughput = gen_len / total_time
    effective_kv_gb = sum(h2o_state.LAYER_KV_SIZES.values()) / 1024**3
    
    # (Latency Breakdown)
    #if h2o_state.PROFILE_STEPS > 0:
    #    print("\n [Latency Breakdown per Step (avg ms)]")
    #    steps = h2o_state.PROFILE_STEPS
    #    stats = h2o_state.PROFILE_STATS
    #    print(f"   Prep (Rotary): {stats['prep']/steps*1000:.4f} ms")
    #    print(f"   Score Calc:    {stats['score']/steps*1000:.4f} ms")
    #    print(f"   Evict (TopK):  {stats['evict']/steps*1000:.4f} ms")
    #    print(f"   Gather/Upd:    {stats['gather']/steps*1000:.4f} ms")
    #    print(f"   Attn Compute:  {stats['attn']/steps*1000:.4f} ms")
    #    print(f"   Total Observed:{stats['total']/steps*1000:.4f} ms")
    #    print("\n")

    # ---------------- Step 2: PPL 测试 ----------------
    print(f"   [Running] {exp_config['name']} - PPL...")
    torch.cuda.empty_cache()
    
    full_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    if full_input_ids.shape[1] < ppl_len:
        print(f"Warning: Text length < PPL Eval Len. Results may be inaccurate.")
        input_ids = full_input_ids
    else:
        input_ids = full_input_ids[:, :ppl_len]
    
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
    
    
    del inputs, input_ids, past_key_values, nlls
    
    if hasattr(model, 'h2o_scores'):
        del model.h2o_scores
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "Experiment": exp_config["name"],
        "Budget (Tokens)": h2o_state.H2O_RECENT_SIZE + h2o_state.H2O_HEAVY_SIZE if h2o_state.ENABLE_KV_COMPRESSION else "Full",
        "KV Size (GB)": effective_kv_gb,
        "TPOT (ms)": tpot,
        "Throughput (T/s)": throughput,
        "PPL": ppl
    }