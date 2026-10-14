import sys, io, pandas as pd
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv(r'C:\fruit_dataset\master_dataset.csv', dtype=str)
active = df[~df['split'].isin(['EXCLUDED','OOD_TEST','']) & (df['domain']!='fruits360')]

sess_info = defaultdict(lambda: {'splits': set(), 'fruits': set(), 'domains': set(), 'labels': set()})
for _, row in active.iterrows():
    sid = row['session_id']
    if sid and row['split']:
        sess_info[sid]['splits'].add(row['split'])
        sess_info[sid]['fruits'].add(row['fruit'])
        sess_info[sid]['domains'].add(row['domain'])
        lbl = row.get('grade_label_normalized','') or row.get('quality_label_normalized','') or row.get('severity_label_normalized','')
        sess_info[sid]['labels'].add(lbl)

leaked = {s: v for s, v in sess_info.items() if 'train' in v['splits'] and 'test' in v['splits']}
print('Sessions in BOTH train AND test:', len(leaked))
for sid, v in list(leaked.items())[:15]:
    fruits_str = ','.join(sorted(v['fruits']))
    labels_str = ','.join(sorted(v['labels']))
    splits_str = ','.join(sorted(v['splits']))
    domains_str = ','.join(sorted(v['domains']))
    print('  sid=%s fruits=%s labels=%s domain=%s splits=%s' % (sid, fruits_str, labels_str, domains_str, splits_str))
