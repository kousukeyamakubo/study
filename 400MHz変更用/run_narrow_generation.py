import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open('learn_data_generator.ipynb', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

cells_to_run = [0,1,2,3,4,5,6,7,8,9,11,14,15,16,17]
nb_sub = nbformat.v4.new_notebook()
nb_sub.cells = [nb.cells[i] for i in cells_to_run]

ep = ExecutePreprocessor(timeout=7200, kernel_name='python3')
ep.preprocess(nb_sub, {'metadata': {'path': 'c:/Users/kosuke/study/400MHz変更用/'}})
print("データ生成完了")
