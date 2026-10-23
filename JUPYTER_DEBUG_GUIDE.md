# 在 Jupyter 中手工调试电影标题 embedding

`export_movie_title_embeddings.py` 是可独立运行的完整导出脚本。
`debug_movie_embeddings.ipynb` 是同一处理流程的未执行调试材料；没有运行结果，
请你自己运行并填写观察记录。

## 启动

在项目根目录的 PowerShell 终端执行：

```powershell
cd D:\code\NCF-MF-embedding
& .venv/Scripts/python.exe -m notebook debug_movie_embeddings.ipynb
```

在浏览器打开的 Notebook 中选择 `NCF-MF (.venv)` 内核。
也可以在 PyCharm 中打开该 `.ipynb`，选择同一个项目环境。
Notebook 第一部分提供 `sys.executable` 检查当前内核使用的解释器。

如果在另一台电脑重建环境：

```powershell
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --sys-prefix --name ncf-mf-embedding --display-name "NCF-MF (.venv)"
```

## 手工运行顺序

1. 导入库，确认工作目录和 `u.item` 文件。
2. 读取电影 ID 和标题，观察原始格式、排序和列表内容。
3. 选择 `mode` 并加载模型。
4. 先对数据集前两部电影，逐个单元格检查分词结果、mask 和 token 向量。
5. 手工观察池化每一步的类型、形状和数值，在预留 Markdown 区记录理解。
6. 完成小批观察后，再运行完整批处理和导出部分。

Shift+Enter 运行当前单元格并移到下一个；Ctrl+Enter 只运行当前单元格。
`for`、`with`、`if/else` 等完整代码块保留在同一个单元格。

Notebook 使用 `Path.cwd()`；如果 `data_path.exists()` 返回 False，
请将 `project_dir` 改为项目绝对路径。Notebook 中不使用 `__file__`。

## 输入与导出

标题来自 `data/ml-100k/u.item`，按原电影 ID 排序，保留年份。
完整导出包含 `movie_ids`、`titles`、`embeddings` 和 `mode`；第 i 行向量对应第 i 个
ID 和标题。BERT-base 的完整向量矩阵预期为 `[1682, 768]`。

- 独立脚本：`outputs/movie_titles_sentence.pt` 或 `outputs/movie_titles_word.pt`。
- 手工 Notebook：`outputs/movie_titles_sentence_notebook.pt` 或 `outputs/movie_titles_word_notebook.pt`。

所有向量文件按项目约定保留在本地 `outputs/`，不提交 Git。
