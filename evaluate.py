from collections import defaultdict
import pandas as pd

src = 'data/asistent/asistent_testset_updated.en-sl.en'
tgt = 'data/asistent/asistent_testset_updated.en-sl.sl'

# create dataset
dd = defaultdict(list)
with open(src) as en, open(tgt) as sl:
    for idx, (src_line, tgt_line) in enumerate(zip(en, sl)):
        dd['en'].append(src_line.strip())
        dd['sl'].append(tgt_line.strip())
df = pd.DataFrame(dd)

results = pd.read_table('parallel-sentences-out-asistent.tsv', header=None,
                        names=['score', 'sl', 'en'])

merged = pd.merge(df, results, how='outer', on='sl')
merged['result'] = merged['en_x'] == merged['en_y']
print(merged['en_x'] == merged['en_y'])