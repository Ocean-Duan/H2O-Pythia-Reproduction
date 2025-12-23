import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 假设 model 已经加载 (如果你没有加载，请取消下面注释重新加载)
# model_id = "EleutherAI/pythia-2.8b"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     device_map="auto", 
#     cache_dir="./model"
# )
# ================= 配置区域 =================
# 建议先用小一点的数据集跑通流程，比如 wikitext-2-raw-v1
DATASET_NAME = "wikitext" 
DATASET_CONFIG = "wikitext-2-raw-v1"
MODEL_ID = "EleutherAI/pythia-2.8b"
CACHE_DIR = "./model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("-" * 40)
print("正在检查 GPTNeoXAttention 实例的具体属性...")
print(f"正在加载模型: {MODEL_ID} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    dtype=torch.float16, 
    device_map="auto", 
    cache_dir=CACHE_DIR
)
# 1. 获取第 0 层的 Attention 对象
# Pythia/GPT-NeoX 的层级结构通常是 model.gpt_neox.layers[i].attention
target_layer = model.gpt_neox.layers[0].attention

# 2. 打印所有属性名 (keys)
print(f"【对象属性列表】 (dir):\n{dir(target_layer)}")
print("-" * 40)
print(f"【实例字典键值】 (__dict__.keys()):\n{target_layer.__dict__.keys()}")

print("-" * 40)
# 3. 针对性检查我们需要的关键属性是否存在，或者换了什么名字
print("关键参数检查:")

# 检查 num_attention_heads
if hasattr(target_layer, "num_attention_heads"):
    print(f"✅ self.num_attention_heads = {target_layer.num_attention_heads}")
else:
    print("❌ self.num_attention_heads 不存在")

# 检查 config 中的 num_attention_heads
if hasattr(target_layer, "config"):
    print("✅ self.config 存在")
    if hasattr(target_layer.config, "num_attention_heads"):
        print(f"   -> self.config.num_attention_heads = {target_layer.config.num_attention_heads}")
    else:
        print("   -> self.config 里也没找到 num_attention_heads")
else:
    print("❌ self.config 不存在")

# 检查 head_size (H2O 算法中也用到了)
if hasattr(target_layer, "head_size"):
    print(f"✅ self.head_size = {target_layer.head_size}")
else:
    print("❌ self.head_size 不存在 (可能需要通过 hidden_size / num_heads 计算)")

# 检查 hidden_size
if hasattr(target_layer, "hidden_size"):
    print(f"✅ self.hidden_size = {target_layer.hidden_size}")
else:
    print("❌ self.hidden_size 不存在")

print("-" * 40)