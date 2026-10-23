# 在 PyCharm 手工调试标题 embedding

脚本：`export_movie_title_embeddings.py`。它独立运行，仅依赖 `torch` 和 `transformers`，
不依赖推荐训练脚本。请自己逐行执行，并记录观察；下面的形状是说明，不是实际调试记录。

## 环境与启动

1. 在 PyCharm 打开项目 `D:\code\NCF-MF-embedding`。
2. 在 Python Interpreter 设置中选择已有解释器：
   `D:\code\NCF-MF-embedding\.venv\Scripts\python.exe`。
3. 打开脚本，在 `titles = ...` 行左侧点击设置断点。
4. 右键脚本，选择 Debug；停住后用 F8 逐行执行自己的脚本。
5. 在 Debug 的 Variables 中观察变量，也可以在 Evaluate Expression 中查看
   `type(input_ids)`、`input_ids.shape`、`input_ids.dtype` 等表达式。

F7 会进入函数内部，Shift+F8 返回调用处；如需理解库函数内部，可手工使用它们。
每次修改标题或 `mode` 后，重新启动调试。

## 逐步观察什么

设 B 为电影数，L 为这个批次补齐后的序列长度，H 为模型隐藏维度（当前 BERT 为 768）。

| 变量 | 类型或形状 | 观察问题 |
|---|---|---|
| `titles` | `list[str]` | 每个元素是否对应一部电影？ |
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

## 可以手工尝试的变化

- 只保留一个标题，观察 B 的变化。
- 输入不同长度的标题，观察 padding 和 mask。
- 把 `mode` 分别设置为 `word`、`sentence`，比较执行路径和输出。
- 调试完成后，在 Python Console 中自行读取导出文件：

```python
import torch
data = torch.load("outputs/movie_titles_sentence.pt", weights_only=True)
data["titles"]
data["embeddings"].shape
```

脚本默认只用已有模型缓存。如果换到未缓存模型的电脑，可以将两处
`local_files_only=True` 改为 `False`，或将 `model_name` 改为完整本地模型目录。

操作参考：[PyCharm 单步调试](https://www.jetbrains.com/help/pycharm/stepping-through-the-program.html)、
[配置 Python 解释器](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)。
