# Word2Vec 标题向量 L2 归一化对照

## 实验改动

对用户提供的每部电影的 100 维标题均值向量，执行
`v_normalized = v / max(||v||_2, 1e-12)`，再通过原来的可训练无偏置
`100 -> 64` 线性层，与电影 ID embedding 相加。归一化发生在电影标题池化之后、
线性投影之前；不改词级池化方式，也不对投影输出或 ID 向量做归一化。
没有增加融合权重或其他可训练参数。

1,682 个输入均为非零向量，L2 长度从平均 26.5576 变为平均 1.0000；归一化后
最小/最大长度约 0.99999982 / 1.00000012，差异来自 float32 精度。
源文件保持原样，归一化向量保存到独立实验输出目录。

原始与 L2 两组使用同一源文件 SHA-256：
`5dd330ad770986c69004befb3714644c16e8faae92bf123e746143a23394f1c7`。
两个版本的输入维度、模型结构、参数量和参数创建顺序相同；归一化不消耗随机数，
每次建模前重新设定 seed 42，因此初始可训练参数一致。

继续采用 sampled-MSE、64 维表示、Adam lr 0.001、weight decay 1e-6、batch 1024、
最多 100 轮和 patience 5。数据划分、每轮负样本与评价候选保持相同。
checkpoint 按最高验证 nDCG@10 选择，测试集在选定后评估一次。

## 归一化前后对比

| 模型与表示 | 最佳 epoch | 实际轮数 | 验证 nDCG@10 | 测试 Recall@10 | 测试 HitRatio@10 | 测试 nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| MF ID 基线 | 5 | 10 | 0.0729 | 0.0890 | 0.3403 | 0.0830 |
| MF + 原始 Word2Vec | 25 | 30 | 0.0528 | 0.0564 | 0.2532 | 0.0526 |
| MF + L2 Word2Vec | 65 | 70 | 0.0992 | 0.1047 | 0.3693 | 0.0917 |
| NCF ID 基线 | 28 | 33 | 0.0949 | 0.0998 | 0.3577 | 0.0867 |
| NCF + 原始 Word2Vec | 8 | 13 | 0.0580 | 0.0796 | 0.3194 | 0.0702 |
| NCF + L2 Word2Vec | 11 | 16 | 0.0823 | 0.0978 | 0.3531 | 0.0842 |

| L2 版本 | 选定训练 MSE | 验证 Recall@10 | 验证 HitRatio@10 | 验证 nDCG@10 |
|---|---:|---:|---:|---:|
| MF | 0.063025 | 0.1307 | 0.3972 | 0.0992 |
| NCF | 0.103938 | 0.1037 | 0.3357 | 0.0823 |

| L2 版本 | 验证 nDCG 相对原始 Word2Vec | 测试 nDCG 相对原始 Word2Vec | 测试 nDCG 相对 ID 基线 |
|---|---:|---:|---:|
| MF | +88.05% | +74.20% | +10.42% |
| NCF | +41.97% | +20.02% | -2.87% |

## 结论与报告纳入规则

在训练开始前说明：以验证 nDCG@10 相较未归一化 Word2Vec 是否改善作为报告纳入
依据，同时对照 ID 基线。两组均满足该条件，因此都加入综合报告，原始结果保留。

- MF：归一化后，三个测试排序指标都超过未归一化版本及 ID 基线。
- NCF：归一化后三个测试排序指标都高于未归一化版本，但仍低于 ID 基线。
- 原始与 L2 的不同训练轮数由相同的早停规则决定；这是完整训练流程的对照，
  不能把效果变化解释成在固定 checkpoint 上仅缩放向量的即时收益。
- 结果支持在当前配置下处理输入向量尺度有帮助。只有 seed 42，尚不能给出
  显著性或跨数据集结论；对 BERT/ID 的比较还存在参数量、初始化及预训练来源差异。
- 本轮根据已观察的历史实验提出；新轮次依旧只按验证集选 checkpoint，
  确认稳健性仍需要多 seed 或独立数据。没有据本轮测试结果继续调参。
- 全部方法按验证 nDCG@10 选择时，MF 和 NCF 当前仍选 BERT 静态词嵌入。

## 复现与文件

```powershell
python benchmark_word2vec_embeddings.py --normalization l2 --epochs 100 --patience 5
python build_bert_report.py charts
# 使用含 reportlab 和 pypdf 的 Python 环境执行下面两步。
python build_bert_report.py pdf
python build_bert_report.py verify
```

- 原始向量实验：`outputs/word2vec_title_mixed_pool_100/`。
- L2 实验：`outputs/word2vec_title_l2_mixed_pool_100/`，包括 86 轮完整历史、最佳模型和实际输入向量。
- L2 结果快照：`results/word2vec_title_l2_mixed_pool_100_summary.json`。
- 综合数据：`results/bert_comparison_report_data.json`，含八组增强模型和两个 ID 基线。
- 完整更新版 PDF：`output/pdf/MF_NCF_BERT_Comparison_Report_L2.pdf`。原 PDF 无法覆盖，因此另存更新版，保留旧文件。
- 图表位于 `output/pdf/figures/`，保留 PNG 与 SVG。
