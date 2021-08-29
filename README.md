# Bitext mining tool
A tool for sentence alignment. Adapted from <https://github.
com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining>

## How to use it
Set variables and correct paths in `bitext_mining.py`.
Filter results with `filter.py`. 

## Easy concatenation of final csv files (optional)

1. Install csvkit:
```
pip install csvkit
```

2. Run:
```
csvstack *.csv  > out.csv
```

## Results
With this method, you can set a threshold that reflects a trade-off between quality and quantity. We released three 
sets with low (1.1), medium (1.2), and high threshold (1.3). The number of pairs in each subset:

1. Low: 496102
2. Medium: 474852
3. High: 425534