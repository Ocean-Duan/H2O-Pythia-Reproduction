# ================= H2O 动态全局状态 =================
# 这些变量会在 main.py 的循环中被修改，并在 model_patch.py 中被读取

ENABLE_KV_COMPRESSION = False
H2O_RECENT_SIZE = 64
H2O_HEAVY_SIZE = 64

LAYER_KV_SIZES = {}

# [新增] 用于存储时间开销分析数据
PROFILE_STATS = {
    "prep": 0.0,    # Rotary & QKV Proj
    "score": 0.0,   # Attention Score Calculation
    "evict": 0.0,   # TopK & Sorting
    "gather": 0.0,  # Gathering KV & Cache Update
    "attn": 0.0,    # Final Attention Computation
    "total": 0.0
}
PROFILE_STEPS = 0

def reset_stats():
    """重置统计数据"""
    global LAYER_KV_SIZES, PROFILE_STATS, PROFILE_STEPS
    LAYER_KV_SIZES = {}
    PROFILE_STATS = {k: 0.0 for k in PROFILE_STATS}
    PROFILE_STEPS = 0