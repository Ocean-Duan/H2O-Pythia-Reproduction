from datasets import load_dataset

def get_real_long_text(min_tokens):
    """
    加载文本，尽可能加载足够多的数据。
    """
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        long_text = ""
        target_char_len = min_tokens * 4 * 2 
        
        for item in dataset["text"]:
            long_text += item
            if len(long_text) > target_char_len: 
                break
        
        print(f"   [Data] Loaded text length: {len(long_text)} chars (Target tokens: {min_tokens})")
        return long_text
    except Exception as e:
        print(f"Dataset load failed: {e}, using dummy text.")
        return "The history of artificial intelligence " * 1000