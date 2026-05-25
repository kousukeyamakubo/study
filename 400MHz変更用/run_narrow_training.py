import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('check.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# セル 0-7（定義・設定・学習）+ 末尾に追加した A/B/C セル（-3, -2, -1）
cells_to_run = list(range(8)) + [len(nb.cells)-3, len(nb.cells)-2, len(nb.cells)-1]
nb_sub = nbformat.v4.new_notebook()
nb_sub.cells = [nb.cells[i] for i in cells_to_run]

ep = ExecutePreprocessor(timeout=7200, kernel_name='python3')
ep.preprocess(nb_sub, {'metadata': {'path': 'c:/Users/kosuke/study/400MHz変更用/'}})
print("学習完了")
