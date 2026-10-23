# 在 PyCharm 手工调试标题 embedding

如果按老师要求使用 Jupyter，请打开 `debug_movie_embeddings.ipynb`，并按
[JUPYTER_DEBUG_GUIDE.md](JUPYTER_DEBUG_GUIDE.md) 操作。Notebook 初始没有执行结果，
供你自己逐个单元格调试；下面保留的是直接调试 `.py` 脚本的方法。

脚本：`export_movie_title_embeddings.py`。它独立运行，仅依赖 `torch` 和 `transformers`，
不依赖推荐训练脚本。请自己逐行执行，并记录观察；下面的形状是说明，不是实际调试记录。

## 环境与启动

1. 在 PyCharm 打开项目 `D:\code\NCF-MF-embedding`。
2. 在 Python Interpreter 设置中选择已有解释器：
   `D:\code\NCF-MF-embedding\.venv\Scripts\python.exe`。
3. 打开脚本，在 `data_path = ...` 行左侧点击设置断点，从读取电影数据开始观察。
4. 右键脚本，选择 Debug；停住后用 F8 逐行执行自己的脚本。
5. 在 Debug 的 Variables 中观察变量，也可以在 Evaluate Expression 中查看
   `type(input_ids)`、`input_ids.shape`、`input_ids.dtype` 等表达式。

F7 会进入函数内部，Shift+F8 返回调用处；如需理解库函数内部，可手工使用它们。
标题来自 MovieLens 100K 的 `data/ml-100k/u.item`，按电影 ID 排序并保留标题中的年份。
每次修改 `mode` 或 `batch_size` 后，重新启动调试。

## 逐步观察什么

设 N 为数据集电影数（MovieLens 100K 为 1,682），B 为当前批次电影数，
L 为当前批次补齐后的序列长度，H 为模型隐藏维度（当前 BERT 为 768）。
默认每批最多 32 部，最后一批可能不足 32 部。

| 变量 | 类型或形状 | 观察问题 |
|---|---|---|
| `records` | `list[list[str]]` | u.item 的第 1、2 列分别是什么？ |
| `movie_ids` | `list[int]`，长度 N | 是否与排序后的电影 ID 对应？ |
| `titles` | `list[str]`，长度 N | 是否保留了原始标题及年份？ |
| `batch_titles` | `list[str]`，长度 B | 当前处理的是哪一批电影？ |
| `inputs` | `BatchEncoding`，行为类似字典 | 有哪些键？分词器为什么返回多个张量？ |
| `input_ids` | `torch.int64`，`[B, L]` | ID 与 `tokens` 中的词元如何对应？ |
| `attention_mask` | `torch.int64`，`[B, L]` | 哪些位置是 padding？ |
| `special_tokens_mask` | `torch.int64`，`[B, L]` | 自动加入的特殊符号位于哪里？ |
| `hidden` | 浮点张量，`[B, L, H]` | 每个 token 为什么都有一个 H 维向量？ |
| `valid_mask` | `torch.bool`，`[B, L]` | 最终有哪些 token 参与池化？ |
| `expanded_mask` | `torch.bool`，`[B, L, 1]` | 为什么多一个维度就能和 hidden 相乘？ |
| `embedding_sum` | 浮点张量，`[B, H]` | `sum(dim=1)` 消去了哪个维度？ |
| `token_counts` | `torch.int64`，`[B, 1]` | 分母为什么不是统一的 L？ |
| `embeddings` | 浮点张量，`[B, H]` | 一行是否对应一个电影标题？ |
| `embedding_batches` | 张量列表 | 已经保存了多少个批次？ |
| `movie_embeddings` | 浮点张量，`[N, H]` | 拼接后是否仍与 movie_ids、titles 一一对应？ |

## 可以手工尝试的变化

- 将 `batch_size` 设置为 1，观察一个电影标题的处理过程。
- 在数据集的一个批次中观察不同长度标题的 padding 和 mask。
- 把 `mode` 分别设置为 `word`、`sentence`，比较执行路径和输出。
- 调试完成后，在 Python Console 中自行读取导出文件：

```python
import torch
data = torch.load("outputs/movie_titles_sentence.pt", weights_only=True)
data["movie_ids"][:3]
data["titles"]
data["embeddings"].shape
```

导出文件中，`data["embeddings"][i]` 对应 `data["movie_ids"][i]` 和
`data["titles"][i]`；电影 ID 与张量行号是两个不同概念。

脚本默认只用已有模型缓存。如果换到未缓存模型的电脑，可以将两处
`local_files_only=True` 改为 `False`，或将 `model_name` 改为完整本地模型目录。

操作参考：[PyCharm 单步调试](https://www.jetbrains.com/help/pycharm/stepping-through-the-program.html)、
[配置 Python 解释器](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)。
