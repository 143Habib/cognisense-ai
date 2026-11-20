from pathlib import Path
import sys
import pandas as pd
from src.multimodal_core import build_dataset
out=Path('data/chunks');out.mkdir(parents=True,exist_ok=True)
subjects=sys.argv[1:]
if not subjects:
 raise SystemExit('Pass participant IDs, e.g. 002 003 004 005')
features,nasa,perf,quality,participants=build_dataset(subjects)
for name,df in [('features',features),('nasa',nasa),('performance',perf),('quality',quality),('participants',participants)]:
 df.to_csv(out / f'{name}_{subjects[0]}_{subjects[-1]}.csv',index=False)
print('Saved',len(features),'trials for',','.join(subjects))
