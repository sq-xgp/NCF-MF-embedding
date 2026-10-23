"""Standalone Transformers example: edit titles and debug each step in PyCharm."""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


if __name__ == "__main__":
    # 1. 在这里修改输入；一个字符串对应一部电影。
    titles = ["Toy Story (1995)", "Jumanji (1995)", "Star Wars (1977)"]
    mode = "sentence"  # "sentence": 完整 BERT；"word": 只查输入词向量表。
    model_name = "bert-base-uncased"  # 也可以填写本地模型文件夹。
    output_path = Path(__file__).resolve().parent / "outputs" / f"movie_titles_{mode}.pt"

    if mode not in {"word", "sentence"}:
        raise ValueError("mode must be word or sentence")
    if not titles or any(not isinstance(title, str) or not title.strip() for title in titles):
        raise ValueError("titles must be a nonempty list of nonempty strings")

    # 2. 本项目已有模型缓存；直接在 CPU 上调试，便于观察变量。
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True, add_pooling_layer=False)
    model.eval()
    model.requires_grad_(False)

    # 3. 将标题转成 token ID，并补齐不同长度的序列。
    inputs = tokenizer(
        titles,
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    special_tokens_mask = inputs.pop("special_tokens_mask")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # 4. 两种方式均得到 [电影数, 序列长度, 768] 的 token 向量。
    with torch.no_grad():
        if mode == "word":
            word_embedding_layer = model.get_input_embeddings()
            hidden = word_embedding_layer(input_ids)
        else:
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

        # 5. 排除自动加入的 [CLS]、[SEP] 和 padding，再取均值。
        valid_mask = attention_mask.bool() & ~special_tokens_mask.bool()
        expanded_mask = valid_mask.unsqueeze(-1)
        masked_hidden = hidden * expanded_mask
        embedding_sum = masked_hidden.sum(dim=1)
        token_counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        embeddings = embedding_sum / token_counts

    # 6. 第 i 行向量始终对应 titles[i]；导出为可重新加载的 .pt 文件。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"titles": titles, "embeddings": embeddings, "mode": mode}, output_path)
    print("First title tokens:", tokens)
    print("input_ids shape:", input_ids.shape)
    print("hidden shape:", hidden.shape)
    print("embeddings shape:", embeddings.shape)
    print("Saved to:", output_path)
