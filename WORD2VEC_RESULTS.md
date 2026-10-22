# 自训练 Word2Vec 直接融合实验

本文保留未归一化实验结果。后续 L2 归一化对照见
[WORD2VEC_L2_RESULTS.md](WORD2VEC_L2_RESULTS.md)，两种版本均已纳入综合 PDF。

## 方法与数据

使用用户提供的 `data/pretrained_word2vec/movielens_title_embeddings.pt`，
包含 1,682 部电影的 100 维标题向量。导出文件说明其聚合方式为：标题中命中词表的
中心词向量取均值，无命中时置零。本次全部电影均有非零向量，词级覆盖为
5,750 / 6,695 = 85.88%；电影覆盖 100% 不代表每个标题词均被覆盖。

加载时依据原始 MovieLens `item_id` 对齐共享 `item_idx`，检查 ID 唯一性、映射和
张量有限值。冻结原始向量，不另做归一化；可训练的无偏置线性层将其从 100 维映射到
64 维，再加到可训练的电影 ID embedding。MF 和 NCF 使用与 BERT 实验相同的模型
结构、训练循环、负采样和评估代码。这里不加载 BERT 权重、不使用 BERT 分词器，
也不执行 Transformer。

训练配置沿用十组实验中按验证 nDCG@10 选出的 sampled-MSE：维度 64，Adam
学习率 0.001，weight decay 1e-6，batch 1024，seed 42，最多 100 轮，patience 5。
按验证 nDCG@10 选 checkpoint；测试集仅在选定后评估一次。时间划分为
79,619 / 9,596 / 10,785，验证与测试分别评估 846 / 861 位用户。

## 最终指标

表中所有测试指标来自同一个验证选定 checkpoint，不分别挑选测试最优值。

| 模型及特征 | 最佳 epoch | 实际轮数 | 验证 nDCG@10 | 测试 Recall@10 | 测试 HitRatio@10 | 测试 nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| MF ID 基线 | 5 | 10 | 0.0729 | 0.0890 | 0.3403 | 0.0830 |
| MF + BERT 静态词嵌入 | 44 | 49 | 0.1045 | 0.1086 | 0.3763 | 0.0976 |
| MF + BERT 上下文编码 | 14 | 19 | 0.0877 | 0.1003 | 0.3566 | 0.0914 |
| MF + Word2Vec | 25 | 30 | 0.0528 | 0.0564 | 0.2532 | 0.0526 |
| NCF ID 基线 | 28 | 33 | 0.0949 | 0.0998 | 0.3577 | 0.0867 |
| NCF + BERT 静态词嵌入 | 46 | 51 | 0.1016 | 0.1023 | 0.3554 | 0.0914 |
| NCF + BERT 上下文编码 | 31 | 36 | 0.1004 | 0.1016 | 0.3554 | 0.0926 |
| NCF + Word2Vec | 8 | 13 | 0.0580 | 0.0796 | 0.3194 | 0.0702 |

Word2Vec 的选定训练 MSE 为 MF 0.088228、NCF 0.120499。相对各自 ID 基线，
测试 nDCG@10 分别变化 -36.61% 和 -19.07%；三个测试排序指标均下降。
本表六组标题特征实验按验证 nDCG@10 选择时，MF 和 NCF 均选择 BERT 静态词嵌入。

## 解释边界

- 这是冻结原始均值向量、线性投影相加、固定超参数、单 seed 的结果，不能推断所有
  Word2Vec 融合方式都会下降。此次没有看到结果后重新调参或挑选更好的测试分数。
- 原始标题向量范数均值 26.56、范围 17.33 至 39.81；没有额外归一化。
  向量尺度可能影响融合及优化，需要独立消融验证，当前结果不足以确定原因。
- Word2Vec 投影增加 6,400 个参数，BERT 投影增加 49,152 个参数。
  MF/NCF Word2Vec 可训练参数总数分别为 177,026 / 184,769。
  各模型的共享 ID 参数未做逐参数初值对齐；参数量与初始化差异也可能影响结果。
- 导出元数据没有充分确定 Word2Vec 预训练语料来源。本次推荐训练按原协议隔离
  验证/测试标签，但不能据此确认整个外部预训练数据链路的独立性。
- 排名候选仍仅为训练期已知电影；测试排除 13 条冷启动正反馈，因此不是冷启动实验。

## 文件与复现

```powershell
python benchmark_word2vec_embeddings.py --epochs 100 --patience 5
python build_bert_report.py charts
# pdf / verify 需要包含 reportlab 和 pypdf 的 Python 环境。
python build_bert_report.py pdf
python build_bert_report.py verify
```

- 原始历史和 checkpoint：`outputs/word2vec_title_mixed_pool_100/`，按项目约定忽略模型产物。
- 精简结果：`results/word2vec_title_mixed_pool_100_summary.json`。
- 完整图表数据：`results/bert_comparison_report_data.json`，现包含八组增强模型（含两组 L2 对照）和两个基线。
- 原始综合 PDF：`output/pdf/MF_NCF_BERT_Comparison_Report.pdf`；包含 L2 对照的完整更新版为 `output/pdf/MF_NCF_BERT_Comparison_Report_L2.pdf`。
- 图表：`output/pdf/figures/`，同时提供 PNG 和 SVG。

使用的输入向量文件 SHA-256：
`5dd330ad770986c69004befb3714644c16e8faae92bf123e746143a23394f1c7`。
如替换用户提供的向量文件，需重新训练；已有报告的指标对应上述输入快照。
