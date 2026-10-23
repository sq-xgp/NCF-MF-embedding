# BERT 上下文标题编码实验

本文记录重写前的历史实验。当前抽取已改用 `transformers`；新版默认输出到独立目录，
本表指标尚未按新版重新训练计算。

## 实现

第二种方式将 MovieLens 的每个电影标题按 BERT uncased WordPiece 规则分词，补上
`[CLS]` 和 `[SEP]` 后输入 `bert-base-uncased` 的 12 层 Transformer。对最后一层
标题 token 的上下文表示做 mean pooling，得到每部电影的 768 维向量；池化时排除
`[CLS]`、`[SEP]` 和 padding。1,682 个标题离线编码一次，向量缓存到输出目录。

BERT 权重保持冻结。MF/NCF 训练时用可训练 768→64 投影层把标题向量映射到推荐
模型维度，再加到电影 ID embedding 上。模型权重来自 Hugging Face Hub 的
[`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased)，
本次使用本机缓存快照 `86b5e0934494bd15c9632b12f734a8a67f723594`。实现使用 PyTorch
复现 BERT encoder 的 embedding、self-attention、残差、LayerNorm 和 GELU 运算，
直接从官方 safetensors 权重载入；不依赖 Transformers 包，也不进行微调。

训练配置选择与静态嵌入相同：在原 10 组比较里，MF、NCF 各自验证集 nDCG@10 最好
的是 sampled-MSE。数据时间划分、负样本、评价候选集和 checkpoint 选择规则均沿用
基线设置；seed 为 42。

## 对比结果

测试指标来自各自验证集 nDCG@10 最优 epoch 的 checkpoint：

| 模型 | 电影标题表示 | 最佳 epoch | 验证 nDCG@10 | 测试 Recall@10 | 测试 HitRatio@10 | 测试 nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| MF | 无标题特征（ID 基线） | 5 | 0.0729 | 0.0890 | 0.3403 | 0.0830 |
| MF | 静态 BERT 词嵌入平均 | 44 | 0.1045 | 0.1086 | 0.3763 | **0.0976** |
| MF | BERT 最后一层上下文向量平均 | 14 | 0.0877 | 0.1003 | 0.3566 | 0.0914 |
| NCF | 无标题特征（ID 基线） | 28 | 0.0949 | 0.0998 | 0.3577 | 0.0867 |
| NCF | 静态 BERT 词嵌入平均 | 46 | 0.1016 | 0.1023 | 0.3554 | 0.0914 |
| NCF | BERT 最后一层上下文向量平均 | 31 | 0.1004 | 0.1016 | 0.3554 | **0.0926** |

在当前单 seed 结果里，静态词向量版 MF 更好；上下文编码版 NCF 略好。相较无标题
特征基线，二者测试 nDCG@10 都有所提高，但一次 seed 不能说明提升稳定。后续应在
多个随机种子下重复，并考虑参数量匹配对照。原排名候选集只含训练期已知电影，因而
这些数字还没有测量标题向量对冷启动电影的排序效果。

基线、静态嵌入和上下文嵌入的精简 JSON 指标快照均保存在 `results/`，例如
[`bert_contextual_title_mixed_pool_100_summary.json`](results/bert_contextual_title_mixed_pool_100_summary.json)。
完整逐轮训练历史、checkpoint 和标题向量保存在被 `.gitignore` 排除的
`outputs/bert_contextual_title_mixed_pool_100/`。

## 复现

```powershell
python benchmark_bert_contextual_embeddings.py --epochs 100 --patience 5
```

需要 MovieLens 100K 本地数据和 `bert-base-uncased` 的 `config.json`、
`model.safetensors`、`vocab.txt`。默认从 Hugging Face 缓存查找，也可用
`--bert-model-dir <快照目录>` 指定。
