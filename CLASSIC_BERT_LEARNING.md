# 从零实现、训练经典 BERT 编码器

入口：`classic_bert_from_scratch.py`。网络用 PyTorch 手写，模型参数随机初始化；
Transformers 仅提供已有的 WordPiece 分词器，不加载预训练模型权重。

当前是第 3 阶段的基础学习示例。等理解结构和训练流程后，再扩展到加载预训练 BERT
进行电影标题微调的第 1 种方案。

## 运行

```powershell
cd D:\code\NCF-MF-embedding
& .venv/Scripts/python.exe classic_bert_from_scratch.py --size tiny --epochs 3
```

脚本使用本地 MovieLens 100K 的 `u.item`，读取 1,682 个电影标题，固定 seed 42，
按标题随机划分 1,514 条训练、168 条验证。它不读取评分或推荐标签。
检测到 CUDA 时使用 GPU，否则使用 CPU。分词器需已缓存，也可以通过
`--tokenizer 本地目录` 指定。

| 配置 | 编码层数 | 隐藏维度 | 注意力头数 | FFN 中间维度 |
|---|---:|---:|---:|---:|
| `--size tiny`，默认学习版 | 2 | 128 | 4 | 512 |
| `--size base`，BERT-base 编码器规格 | 12 | 768 | 12 | 3072 |

两种配置的核心结构相同，均有 512 个可学习的位置向量、两种 segment 向量、
双向注意力、残差连接、post-LayerNorm、GELU 和 Dropout。
base 参数更多、占用更多显存；先使用 tiny 理解代码。

## 按顺序读这些类

1. `BertConfig`：保存词表大小、层数、向量维度等配置。
2. `BertEmbeddings`：token、位置、segment 三种向量相加，再归一化和 Dropout。
3. `MultiHeadSelfAttention`：手写 Q、K、V、多头拆分、缩放点积、softmax 和输出投影。
4. `BertLayer`：注意力、残差与 LayerNorm，再进行前馈网络、残差与 LayerNorm。
5. `ClassicBert`：堆叠编码层，输出每个 token 的上下文向量。
6. `ClassicBertForMLM`：增加词预测头，输出每个位置对应词表各词的分数。
7. `mask_tokens` 和 `main`：准备训练目标，并执行前向、反向传播和参数更新。

编码器输出与之前学过的 `hidden` 对应。tiny 的形状是 `[B, L, 128]`，base 是
`[B, L, 768]`；MLM 预测头的输出为 `[B, L, vocab_size]`。

## 这次训练什么

任务是 MLM（掩码词预测）：随机选择约 15% 的标题内容 token，预测它们原来的编号。
被选位置中，80% 替换成 `[MASK]`，10% 替换为随机词，10% 保留原词；其他位置的
label 为 `-100`，不计入交叉熵。极短批次若未选中任何位置，会补选一个有效位置，
防止整批没有监督信号。

标题是独立文本，因此训练示例采用 MLM；原始 BERT 预训练中的 NSP（下一句预测）
不在此次训练目标内。这里没有把相邻电影 ID 当作自然语言上下句。

训练的关键步骤直接写在循环中：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(**inputs)
loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

训练 mask 每批重采样；验证 mask 生成后保持固定，便于逐轮对照。每轮记录训练损失、
验证损失和验证掩码词准确率，按最低验证损失保存模型。
这些指标不等同于 MF/NCF 的推荐排序指标。

## 输出与重新加载

默认目录 `outputs/classic_bert_scratch_tiny/` 中包含：

- `best_model.pt`：最低验证损失对应的参数、配置和 epoch。
- `metrics.json`：训练前验证值、逐轮记录、参数更新检查。
- `tokenizer/`：训练时使用的分词器。

```python
import torch
from classic_bert_from_scratch import BertConfig, ClassicBertForMLM

checkpoint = torch.load("outputs/classic_bert_scratch_tiny/best_model.pt", map_location="cpu", weights_only=True)
model = ClassicBertForMLM(BertConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

`model.bert(input_ids, attention_mask, token_type_ids)` 返回 token 向量。
之后可以沿用已学过的有效 token 均值池化。

这份小语料训练用于理解结构和参数更新，不能代替大规模语言预训练。

学习版已在本地 GPU 实际训练 3 轮：验证 MLM 损失从训练前的 10.3525 降至
5.8113，参数更新和模型重新加载检查通过。完整记录见
`results/classic_bert_scratch_tiny_summary.json`。离线数值测试还核对了手写模型与
相同权重的 Transformers BERT 的输出，以及短序列的掩码与反向传播。

结构参考：[BERT 原始论文](https://aclanthology.org/N19-1423/)。
