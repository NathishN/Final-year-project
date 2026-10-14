import pandas as pd

df = pd.read_csv(r'C:\fruit_dataset\outputs\casm_v2\phase50_split_v2_manifest.csv')
cross = df.groupby('session_id')['split'].nunique()
leaked = cross[cross > 1].index.tolist()
print('Sessions spanning multiple splits:', len(leaked))
for s in leaked[:5]:
    rows = df[df['session_id'] == s]
    sp = rows['split'].value_counts().to_dict()
    fruit = rows['fruit'].iloc[0] if 'fruit' in rows.columns else 'N/A'
    grade = rows['severity_grade'].iloc[0]
    n = len(rows)
    print(f'  session={s}: splits={sp}, n={n}, fruit={fruit}, grade={grade}')

# Also check banana grade 2 session
if 'fruit' in df.columns:
    bg2 = df[(df['fruit'].str.lower() == 'banana') & (df['severity_grade'] == 2)]
    print()
    print('Banana Grade2 sessions:')
    for sess, grp in bg2.groupby('session_id'):
        print(f'  {sess}: {grp["split"].value_counts().to_dict()}, n={len(grp)}')
