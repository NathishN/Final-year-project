import sys, io, os, re, pandas as pd
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CSV_PATH = r'C:\fruit_dataset\master_dataset.csv'
df = pd.read_csv(CSV_PATH, dtype=str)
print(f'Loaded {len(df):,} rows\n')

# Check 1
bad = df['image_path'].str.contains('100 x 100|100x100', regex=True, na=False)
status = 'PASS' if bad.sum() == 0 else 'FAIL'
print(f'Check 1 - Forbidden 100x100 folders: {bad.sum()} rows -> [{status}]')

# Check 2
excl = df[df['split'] == 'EXCLUDED']
non_b = excl[~excl['image_path'].str.contains('Banana_3rd_grade', na=False)]
status = 'PASS' if len(non_b) == 0 else 'FAIL'
print(f'Check 2 - EXCLUDED rows: {len(excl)} total, {len(non_b)} non-Banana -> [{status}]')

# Check 3
active = df[~df['split'].isin(['EXCLUDED', 'OOD_TEST', '']) & (df['domain'] != 'fruits360')]
sess_splits = defaultdict(set)
for _, row in active.iterrows():
    if row['session_id'] and row['split']:
        sess_splits[row['session_id']].add(row['split'])
leaks = sum(1 for s, splits in sess_splits.items() if 'train' in splits and 'test' in splits)
status = 'PASS' if leaks == 0 else 'FAIL'
print(f'Check 3 - Session leakage (train/test): {leaks} leaks -> [{status}]')

# Check 4
active2 = df[df['split'] != 'EXCLUDED']
print('\nCheck 4 - Split distribution:')
for sp, n in active2['split'].value_counts().sort_index().items():
    print(f'  {sp:<10}: {n:>7,}  ({n/len(active2)*100:.1f}%)')

# Check 5
print('\nCheck 5 - Class balance:')
for label_type, col in [
    ('grade',    'grade_label_normalized'),
    ('quality',  'quality_label_normalized'),
    ('severity', 'severity_label_normalized'),
]:
    sub = df[(df['split'].isin(['train', 'val', 'test'])) & df[col].notna() & (df[col].str.strip() != '')]
    if len(sub) == 0:
        continue
    print(f'  [{label_type.upper()}]')
    for fruit, grp in sub.groupby('fruit'):
        counts = grp[col].value_counts().sort_index()
        if len(counts) < 2:
            continue
        ratio = round(counts.max() / max(counts.min(), 1), 1)
        flag = ' [SEVERE]' if ratio > 5 else (' [moderate]' if ratio > 2 else '')
        info = ' | '.join(f'{k}={v}' for k, v in counts.items())
        print(f'    {fruit:<8}: {info}  ratio={ratio}x{flag}')

# Check 6
print('\nCheck 6 - Per-fruit row counts by split:')
active3 = df[~df['split'].isin(['EXCLUDED'])]
pivot = pd.crosstab(active3['fruit'], active3['split'])
print(pivot.to_string())

# Check 7
print('\nCheck 7 - Domain x Split:')
pivot2 = pd.crosstab(active3['domain'], active3['split'])
print(pivot2.to_string())

# Check 8
import random
random.seed(42)
sample_rows = df.sample(min(300, len(df)), random_state=42)
missing = [row['image_path'] for _, row in sample_rows.iterrows() if not os.path.exists(row['image_path'])]
status = 'PASS' if len(missing) == 0 else 'FAIL'
print(f'\nCheck 8 - File existence (300 sample): {len(missing)} missing -> [{status}]')

print('\nVerification complete.')
