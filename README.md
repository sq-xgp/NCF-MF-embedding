# NCF-MF-embedding：MovieLens 100K 推荐模型实验基础

本目录从此前的 MF / NCF 实验代码整理而来，作为后续用户与物品 embedding
表征实验的起点。`baselines.py` 提供共享数据入口与 MF/NCF 模型工厂；原训练和
评估脚本保持不变，可直接复现实验。

```python
from baselines import build_model, load_experiment_data

data = load_experiment_data()
mf = build_model("mf", "ranking", data.num_users, data.num_items, embedding_dim=64)
ncf = build_model("ncf", "ranking", data.num_users, data.num_items, embedding_dim=64)
```

`task="rating"` 使用 `(rating-1)/4` 标签和 `[0,1]` 预测值；`task="ranking"`
使用 4–5 分为正、1–2 分为负的二值协议，模型输出未经过 sigmoid 的 logit。
两套任务不能混用标签或重复施加 sigmoid。评分评价使用训练期已知电影，排序
评价则沿用既有候选集和 Recall@K、HitRatio@K、nDCG@K 实现。

本机已有的 MovieLens `u.data`、`u.user`、`u.item` 已复制到 `data/ml-100k/`，
未重新下载或重新随机划分；这些原始数据被 `.gitignore` 排除，不能随代码再分发。

## 已迁移的 10 组正式实验结果

`outputs/ranking_benchmark_mixed_pool_100/` 保存此前 100 epoch 上限的 MF/NCF
十组二值排序实验。`summary.json` 含十组对比及评估协议；每组子目录中的
`metrics.json` 含逐 epoch 的训练损失与验证 Recall@10、HitRatio@10、nDCG@10，
`best_model.pt` 是按验证 nDCG@10 选出的权重。该实验启用 patience=5 的早停，
因此各组实际历史轮数不一定达到 100。部分子目录另有用于续训的 `progress.pt`。

这批结果采用“训练集真实负样本 + 训练期未评分电影”的混合负采样池；旧版
`ranking_benchmark_60`、`ranking_benchmark_1000` 使用不同的采样协议，不能混作
同一组对照。所有 `outputs/` 文件均为本机实验产物，已被 `.gitignore` 排除。

## 原实验说明

这是一个用于多版本公平对比的显式评分基线。两座塔分别是用户 ID embedding 和电影 ID embedding，二者点积后通过 sigmoid 输出归一化评分。

本仓库也包含 MF / NCF 的二值反馈 10 组排名基准、两种负采样池实验、
100-epoch 上限复现脚本，以及用户 / 物品侧表征消融。使用 Python 3.10+：

```powershell
python -m pip install -r requirements.txt
```

MovieLens 100K 原始数据**不在仓库中**：随数据提供的许可禁止未经单独许可再分发。
请从 [GroupLens 官方页面](https://grouplens.org/datasets/movielens/100k/)自行获取，
并按 [data/README.md](data/README.md) 放置 `u.data`、`u.user`、`u.item`。
项目运行时不会自动下载，也不会重新随机划分数据。

## 已固定的实验协议

- 数据：MovieLens 100K 的 `u.data`，共 100,000 条评分。
- 标签：`y = (rating - 1) / 4`，把 1--5 精确映射到 0--1。
- 划分：每位用户内部按 `(timestamp, item_id)` 稳定排序，前 80% 训练、接下来 10% 验证、其余测试，边界向下取整。
- 模型：`sigmoid(user_embedding · item_embedding)`。
- 损失：归一化评分上的 MSE。
- 模型选择：验证集原始 1--5 尺度 RMSE 最低的 checkpoint。
- 最终指标：测试集原始尺度 RMSE 和 MAE。
- 冷启动：验证/测试中训练未见的电影不进入基础 MF 主指标，并在结果中单独计数。
- 随机性：固定种子 42；训练集仅在 epoch 内打乱。

## 路径

```text
data/ml-100k/u.data   # 原始评分
train.py              # 数据、模型、训练和评估
recommend.py          # 从训练期已知且用户未评分的电影中生成 Top-K
tests/test_mf.py      # 关键协议测试
outputs/best_model.pt # 最佳验证集模型
outputs/metrics.json  # 配置、划分统计和指标
```

## 运行

```powershell
python -m unittest discover -s tests -v
python train.py
python recommend.py --user-id 1 --top-k 10
python evaluate_ranking.py --checkpoint outputs/biased_tuned/best_model.pt --k 10
```

BPR V3 使用相同时间划分，以验证集 NDCG@10 选择模型：

```powershell
python train_bpr.py
```

最简 NCF 单塔把用户与电影 embedding 拼接后输入 MLP，继续使用相同的显式评分协议：

```powershell
python train_ncf.py
```

其默认张量流为 `[batch] -> [batch,32] -> concat [batch,64] -> MLP -> [batch]`，
输出保存在 `outputs/ncf/`。模型以验证集原始 1--5 尺度 RMSE 选择 checkpoint，
测试 RMSE/MAE 只在最终选定模型上计算。

混合 MF 在 MF V2 上加入 MovieLens 自带侧信息：用户年龄、性别、职业，以及电影的
19 维类型和上映年份。验证集选择的最终配置使用两侧特征和 `weight_decay=1e-5`：

```powershell
python train_hybrid_mf.py --feature-set both --weight-decay 0.00001
```

用户与电影侧特征分别投影到 32 维，通过“用户内容 × 电影 ID”“用户 ID × 电影内容”
和“用户内容 × 电影内容”三个交叉项增强原有带偏置 MF。正式结果保存在
`outputs/hybrid_mf/`；特征及超参数消融保存在 `work/hybrid_tuning/`，且均使用
`--validation-only`，没有用测试集调参。

## 二值反馈排名基准（10 组）

`benchmark_ranking_losses.py` 在相同的 MovieLens 100K 时间划分上，对 MF 和单塔 NCF
分别比较 5 种训练设置：全部真实反馈的 MSE/BCE，以及一比一随机负采样的
MSE/BCE/BPR。4–5 分作为正样本，1–2 分作为显式负样本，3 分不提供训练标签。
当前随机负样本池是“该用户训练集中的 1–2 分真实负样本”与“该用户训练期未评分、
但全局在训练集中出现过的电影”的并集；从并集中均匀、有放回地抽取。未评分不等于
真实不喜欢；验证集和测试集评分不参与构池。每个 epoch 对每个正样本重新抽取一个负样本，所有模型和损失使用
同一个 epoch 对应的随机种子，因此采样对齐。

新采样池的 100 轮上限实验请写入独立目录，旧的 `ranking_benchmark_60` 和
`ranking_benchmark_1000` 结果仍属于“仅采未评分”的旧协议，不可混作同一实验：

```powershell
python benchmark_ranking_losses.py --epochs 100 --output-dir outputs/ranking_benchmark_mixed_pool_100
```

统一配置：embedding 维度 64、Adam 学习率 0.001、weight decay 1e-6、
batch size 1024、随机种子 42、最多 100 epoch、patience 5。按验证集 nDCG@10
选择 checkpoint，最终计算测试 Recall@10、HitRatio@10 和 nDCG@10。
评价使用完整的训练期已知电影候选集，排除用户此前交互；HitRatio@10 是
Top-10 至少命中一个正样本的用户比例。正式十组结果写入
`outputs/ranking_benchmark_mixed_pool_100/summary.json`，每组还保存独立 checkpoint、训练曲线
和指标文件。

若要用当前新协议做完整 1000 轮学习曲线实验，应使用新的输出目录（不能续训
旧协议的 `ranking_benchmark_1000`）；`--full-epochs` 会关闭早停。每个 epoch
计算并保存验证集 Recall@10、HitRatio@10、nDCG@10，测试集仍只在选出
验证最优 checkpoint 后评价一次。长实验每 20 轮保存可恢复进度：

```powershell
python benchmark_ranking_losses.py --epochs 1000 --full-epochs --resume --output-dir outputs/ranking_benchmark_mixed_pool_1000
python summarize_ranking_1000.py --input-dir outputs/ranking_benchmark_mixed_pool_1000
python plot_ranking_benchmark.py --input-dir outputs/ranking_benchmark_mixed_pool_1000 --output-dir outputs/ranking_benchmark_mixed_pool_1000
```

## 排名任务的用户 / 物品侧表征消融

先按旧版仅未评分采样与新版并集采样结果中的验证集 nDCG@10 选择对照组：
MF + BPR + 训练期未评分电影负采样。`benchmark_side_features.py` 固定这套训练与
评价方式，分别训练无侧特征、用户侧、物品侧和双侧四组。用户侧使用年龄、性别、
职业；物品侧使用电影类型、上映年份。侧特征线性投影加到原 MF 向量中，投影
初始为零，确保四组起点与纯 MF 对照一致。每组最多 100 轮，patience 5。

```powershell
python benchmark_side_features.py --epochs 100 --output-dir outputs/side_features_mf_bpr_100
```

结果位于 `outputs/side_features_mf_bpr_100/summary.json`，每组目录包含逐轮
`metrics.json` 和最佳模型。旧的“仅未评分”协议在此仅用于对照，不改变前一节
十组实验的并集采样设置。

## BERT 静态词嵌入排名实验

`benchmark_bert_static_embeddings.py` 在 10 组排名对比中分别选取验证集 nDCG@10
最高的 MF 和 NCF 配置（两者均为并集负采样 + MSE），将电影标题切成 uncased
WordPiece，直接查询 `bert-base-uncased` 的输入词嵌入表，并对标题 token 向量取平均。
此方案不运行 BERT Transformer 层，也不微调 BERT；每个标题向量再经可训练线性层
映射到 64 维，与电影 ID embedding 相加。BERT safetensors 和 `vocab.txt` 需位于
Hugging Face 缓存目录，或通过 `--bert-model-dir` 指向本地模型快照。

```powershell
python benchmark_bert_static_embeddings.py --epochs 100 --patience 5
```

逐模型指标、最佳 checkpoint 和预计算标题向量保存在
`outputs/bert_static_word_embedding_mixed_pool_100/`。输出被 `.gitignore` 排除；
已汇总的对照结果见 [BERT_STATIC_RESULTS.md](BERT_STATIC_RESULTS.md)。

第二种方案把完整电影标题（含 `[CLS]`、`[SEP]`）输入 BERT 的 12 层 Transformer，
对最后一层的标题 token 向量做 mean pooling。编码器冻结，只离线编码 1,682 个标题
一次；后续推荐模型训练沿用相同的 sampled-MSE 协议：

```powershell
python benchmark_bert_contextual_embeddings.py --epochs 100 --patience 5
```

逐模型指标、checkpoint 和标题向量保存在
`outputs/bert_contextual_title_mixed_pool_100/`。对比说明见
[BERT_CONTEXTUAL_RESULTS.md](BERT_CONTEXTUAL_RESULTS.md)。

## 自训练 Word2Vec 直接融合

使用 `data/pretrained_word2vec/movielens_title_embeddings.pt` 中的 100 维电影标题
向量，按原电影 ID 对齐、冻结，并通过可训练的 100→64 线性层加到电影 ID 向量。
MF 和 NCF 继续使用上述 sampled-MSE 协议；不运行 BERT。

```powershell
python benchmark_word2vec_embeddings.py --epochs 100 --patience 5
# L2 对照：逐电影归一化已池化的 100 维标题向量，再投影并与 ID 相加。
python benchmark_word2vec_embeddings.py --normalization l2 --epochs 100 --patience 5
```

结果和解释见 [WORD2VEC_RESULTS.md](WORD2VEC_RESULTS.md)。完整日志与最佳模型位于
`outputs/word2vec_title_mixed_pool_100/`；版本管理的结果快照位于
`results/word2vec_title_mixed_pool_100_summary.json`。

L2 对照使用独立目录 `outputs/word2vec_title_l2_mixed_pool_100/`，结果见
[WORD2VEC_L2_RESULTS.md](WORD2VEC_L2_RESULTS.md)。原始与 L2 两组保持相同架构、
初始化、采样与训练配置；MF 归一化后的测试 nDCG 高于 ID 基线，NCF 改善后仍低于基线。

综合报告更新版为 `output/pdf/MF_NCF_BERT_Comparison_Report_L2.pdf`，扩充为
两种 BERT 方式、原始及 L2 Word2Vec 的八组增强模型、两个 ID 基线、最终对比表格与逐轮曲线。
原 13 页报告文件保留。
生成脚本读取已有日志，不重新训练：

```powershell
python build_bert_report.py charts
# 以下两步需安装 reportlab 和 pypdf，或使用包含它们的 Python 环境。
python build_bert_report.py pdf
python build_bert_report.py verify
```

Embedding Case Study（默认 user 1 与 Star Wars）：

```powershell
python analyze_embeddings.py --user-id 1 --item-id 50 --top-k 5
```

其负采样协议固定为：4--5 分为正样本；1--2 分为明确负样本；3 分中性；每个正样本以 50% 概率采样明确负样本，否则采样训练期未评分电影。

带偏置的 V2 使用相同协议训练到独立目录：

```powershell
python train.py --use-bias --output-dir outputs/biased
python recommend.py --user-id 1 --top-k 10 --use-bias --checkpoint outputs/biased/best_model.pt
```

调参阶段必须隐藏测试指标，例如：

```powershell
python train.py --use-bias --weight-decay 0.0001 --validation-only --output-dir work/tuning/wd-1e-4
```

快速冒烟测试可以减少训练轮数：

```powershell
python train.py --epochs 2 --output-dir work/smoke-output
```

不要根据测试集结果选择 epoch 或调整超参数。所有版本必须使用相同的数据、标签公式、划分函数及冷启动规则。
