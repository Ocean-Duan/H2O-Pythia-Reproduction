import torch

# ================= 静态配置区域 =================
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "./RankKV/model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 实验长度设置
BENCHMARK_CONFIG = {
    "load_text_min_tokens": 2048,
    "speed_test_prompt_len": 512,
    "speed_test_gen_len": 1024,
    "ppl_eval_seq_len": 2048
}