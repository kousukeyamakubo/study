import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('check.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# セル 0-7（定義）+ 末尾5セル（狭角度設定・分割・学習・シナリオ評価・CSV出力）
cells_to_run = list(range(8)) + list(range(len(nb.cells)-5, len(nb.cells)))
nb_sub = nbformat.v4.new_notebook()
nb_sub.cells = [nb.cells[i] for i in cells_to_run]

ep = ExecutePreprocessor(timeout=7200, kernel_name='python3')
ep.preprocess(nb_sub, {'metadata': {'path': 'c:/Users/kosuke/study/400MHz変更用/'}})
print("シナリオテスト完了")
