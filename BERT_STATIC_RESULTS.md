# BERT 静态词嵌入实验结果

## 方法

从 10 组基线中，分别按验证集 nDCG@10 选择 MF 和 NCF 的最佳训练配置；两者均为
并集负采样 + MSE。基线 checkpoint 的选择规则、时间划分、负采样方式和排名评价
保持不变。对每个电影标题，使用 `bert-base-uncased` tokenizer 的 WordPiece 切分，
从预训练的输入词嵌入表查出各 token 向量并取算术平均。BERT 编码器未运行，也没有
微调。标题向量经可训练的 768→64 线性层投影，再与可训练的电影 ID 向量相加。

这对应直接使用 BERT 的**静态词嵌入**，不是把标题输入 Transformer 后得到的上下文
向量。模型来源为 Hugging Face Hub 的
[`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)，
本次从本机缓存快照 `86b5e0934494bd15c9632b12f734a8a67f723594` 读取词嵌入表；
快照标识保存在对应的结果 JSON 中。

## 指标对照

各行均为验证集 nDCG@10 选出的 checkpoint 在测试集上的一次性指标：

| 模型 | 特征 | 最佳 epoch | 验证 nDCG@10 | 测试 Recall@10 | 测试 HitRatio@10 | 测试 nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| MF | ID 基线 | 5 | 0.0729 | 0.0890 | 0.3403 | 0.0830 |
| MF | ID + BERT 静态词嵌入 | 44 | 0.1045 | 0.1086 | 0.3763 | 0.0976 |
| NCF | ID 基线 | 28 | 0.0949 | 0.0998 | 0.3577 | 0.0867 |
| NCF | ID + BERT 静态词嵌入 | 46 | 0.1016 | 0.1023 | 0.3554 | 0.0914 |

测试 nDCG@10 相比各自基线，MF 提高约 17.5%，NCF 提高约 5.3%。本实验固定单个
随机种子 42，属于初步对照；还应使用多个种子复跑，确认提升是否稳定。标题投影层
给每个版本额外增加 49,152 个可训练参数，因此后续也可加入参数量匹配的对照。

精简指标快照已纳入 Git：
[`results/ranking_benchmark_mixed_pool_100_summary.json`](results/ranking_benchmark_mixed_pool_100_summary.json)
和
[`results/bert_static_word_embedding_mixed_pool_100_summary.json`](results/bert_static_word_embedding_mixed_pool_100_summary.json)。
原始逐轮历史、checkpoint 和 `.pt` 标题向量仍保存在被 `.gitignore` 排除的
`outputs/` 中。

## 复现

```powershell
python benchmark_bert_static_embeddings.py --epochs 100 --patience 5
```

需要 MovieLens 100K 本地数据和 `bert-base-uncased` 模型快照。默认从 Hugging Face
缓存查找模型；也可传入 `--bert-model-dir <快照目录>`。
