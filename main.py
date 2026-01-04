import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from src import config, enable_h2o_monkey_patch, get_real_long_text, run_benchmark

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    print(f"Config: {config.BENCHMARK_CONFIG}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, cache_dir=config.CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, 
        dtype=torch.float16, 
        device_map="auto", 
        cache_dir=config.CACHE_DIR
    )
    
    enable_h2o_monkey_patch(model)
    
    long_text = get_real_long_text(min_tokens=config.BENCHMARK_CONFIG["load_text_min_tokens"])

    # ================= 定义实验组 =================
    all_experiments = [
        {"name": "Baseline", "compress": False, "r": 0, "h": 0},
        #{"name": "H2O (64+64)", "compress": True, "r": 64, "h": 64},
        #{"name": "Local (128)", "compress": True, "r": 128, "h": 0},
        {"name": "H2O (128+128)", "compress": True, "r": 128, "h": 128},
        {"name": "H2O (64+192)", "compress": True, "r": 64, "h": 192},
        #{"name": "H2O (32+224)", "compress": True, "r": 32, "h": 224},
        {"name": "Local (256)", "compress": True, "r": 256, "h": 0},
        #{"name": "H2O (256+256)", "compress": True, "r": 256, "h": 256},
        #{"name": "H2O (128+384)", "compress": True, "r": 128, "h": 384},
        #{"name": "H2O (64+448)", "compress": True, "r": 64, "h": 448},
    ]

    results = []
    print("\n>>> Starting Full NeurIPS Experiments Suite...\n")
    
    for exp in all_experiments:
        res = run_benchmark(model, tokenizer, long_text, exp)
        results.append(res)
        print(f"   -> Done. PPL: {res['PPL']:.2f}, KV: {res['KV Size (GB)']:.4f} GB")

    df = pd.DataFrame(results)
    print("\n================ FINAL PAPER RESULTS ================")
    print(df.to_markdown(index=False))
    df.to_csv("personal_results.csv", index=False)