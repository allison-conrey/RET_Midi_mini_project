# forced_rank.py
import numpy as np
import midi2audio
from midi2audio import FluidSynth
import pandas as pd
from random import shuffle

fs = FluidSynth()
filenames = [f'{i}.mid' for i in range(10)]
rows = list(range(10))
cols = list(range(10))

df_rank = (.5 * np.ones((10, 10))).tolist()
df_rank = pd.DataFrame([], index=rows, columns=cols)
scores = []

for r in rows:
    for c in cols:
        if df_rank.idx[r, c].isna():
            f1 = f'{r}.mid'
            f2 = f'{c}.mid'
            for i in [r, c]:
                fn = f'{r}.mid'
                print(fn)
                fs.play_midi(fn)
                song_scores = dict(first=r, second=c, song_index=i, score=3)
                ans = input('Rate song 0-5 (0 bad, 5 good, [3] ok): ')
                try:
                    song_scores['score'] = float(ans)
                except (ValueError, TypeError):
                    song_scores['score'] = ans
                scores.append(song_scores)
            ans = input('Was second song better? ')
            try:
                df_rank.idx[r, c] = float(ans)
            except (ValueError, TypeError):
                df_rank.idx[r, c] = ans
