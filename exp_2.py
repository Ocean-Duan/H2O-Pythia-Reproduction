# 实验配置列表
from run import *

experiments = [
    {"name": "Full Cache", "method": "full", "recent": 0, "heavy": 0},
    {"name": "H2O (50% budget)", "method": "h2o", "recent": 128, "heavy": 128},
    {"name": "H2O (20% budget)", "method": "h2o", "recent": 50, "heavy": 50},
    {"name": "Local Only (Baseline)", "method": "h2o", "recent": 256, "heavy": 0}, 
]

results = []

for exp in experiments:
    # 1. 重新加载模型 (清除之前的 KV Cache 和 钩子)
    model = ... 
    
    # 2. 根据 exp 配置 Monkey Patch
    if exp["method"] == "h2o":
        # 修改全局变量或传入参数来控制 H2O_RECENT_SIZE 和 H2O_HEAVY_SIZE
        pass 
        
    # 3. 运行 PPL 测试
    ppl = calculate_ppl(...)
    
    # 4. 运行显存/速度测试
    max_mem, avg_tpot = run_generation_benchmark(...)
    
    results.append({**exp, "ppl": ppl, "memory": max_mem, "tpot": avg_tpot})

# 5. 保存结果到 CSV 以便画图