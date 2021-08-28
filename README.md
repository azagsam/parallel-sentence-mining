# Bitext mining tool
A tool for sentence alignment. Adapted from https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining

##### How to use it
Set variables and correct paths in `bitext_mining.py`

##### Easy concatenation of final csv files (optional)

1. Install csvkit:
```
pip install csvkit
```

2. Run:
```
csvstack *.csv  > out.csv
```