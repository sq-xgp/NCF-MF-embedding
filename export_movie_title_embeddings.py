"""Read MovieLens 100K titles and export BERT embeddings; debug in PyCharm."""

import csv
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


if __name__ == "__main__":
    # 1. 从 MovieLens 100K 的电影文件读取 ID 和原始标题（保留年份）。
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "ml-100k" / "u.item"
    with data_path.open("r", encoding="latin-1", newline="") as file:
        reader = csv.reader(file, delimiter="|", quoting=csv.QUOTE_NONE)
        records = list(reader)
    records.sort(key=lambda row: int(row[0]))
    movie_ids = [int(row[0]) for row in records]
    titles = [row[1] for row in records]

    mode = "sentence"  # "sentence": 完整 BERT；"word": 只查输入词向量表。
    model_name = "bert-base-uncased"  # 也可以填写本地模型文件夹。
    batch_size = 32
    output_path = project_dir / "outputs" / f"movie_titles_{mode}.pt"

    if mode not in {"word", "sentence"}:
        raise ValueError("mode must be word or sentence")
    if not titles or any(not isinstance(title, str) or not title.strip() for title in titles):
        raise ValueError("titles must be a nonempty list of nonempty strings")
    if batch_size <= 0 or len(set(movie_ids)) != len(movie_ids):
        raise ValueError("batch_size must be positive and movie IDs must be unique")

    # 2. 本项目已有模型缓存；直接在 CPU 上调试，便于观察变量。
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True, add_pooling_layer=False)
    model.eval()
    model.requires_grad_(False)

    # 3. 分批处理所有电影，避免一次将整个数据集送入 BERT。
    embedding_batches = []
    with torch.no_grad():
        for start in range(0, len(titles), batch_size):
            batch_titles = titles[start:start + batch_size]
            inputs = tokenizer(
                batch_titles,
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

            # 4. 得到 [批次电影数, 序列长度, 768] 的 token 向量。
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
            embedding_batches.append(embeddings)

    # 6. 按原顺序拼接；第 i 行对应 movie_ids[i] 和 titles[i]。
    movie_embeddings = torch.cat(embedding_batches, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"movie_ids": movie_ids, "titles": titles, "embeddings": movie_embeddings, "mode": mode}, output_path)
    print("Movie count:", len(movie_ids))
    print("embeddings shape:", movie_embeddings.shape)
    print("Saved to:", output_path)
