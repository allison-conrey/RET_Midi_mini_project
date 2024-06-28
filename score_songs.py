# forced_rank.py
from pathlib import Path
from random import shuffle

import numpy as np
# import midi2audio
import pandas as pd
from midi2audio import FluidSynth
import jsonlines

fs = FluidSynth()

try:
    DATA_DIR = Path(__file__).parent / 'data'
except Exception:
    DATA_DIR = Path.cwd().parent / 'data'
assert DATA_DIR.is_dir()

GEN_DIR = DATA_DIR / 'midi_files' / '100epochs'
assert GEN_DIR.is_dir()

df_rank = (.5 * np.ones((10, 10))).tolist()
scores = []

paths = list(GEN_DIR.glob('*.mid'))
shuffle(paths)
filenames = [p.name for p in paths]

assert len(paths) >= 10

try:
    with jsonlines.open(GEN_DIR / 'song_scores.jsonlines', mode='a') as fout:
        for r, path1 in enumerate(paths):
            for c, path2 in enumerate(paths):
                if df_rank[r][c] == .5:
                    pair_scores = {}
                    for i, pth in enumerate([path1, path2]):
                        filenum = pth.name.split('.')[0]
                        print(pth)
                        fs.play_midi(pth)
                        pair_scores[filenum] = 3
                        ans = input('Rate song 0-5 (0 bad, 5 good, [3] ok): ')
                        try:
                            pair_scores[filenum] = float(ans)
                        except (ValueError, TypeError):
                            pair_scores[filenum] = ans or 3
                    ans = input('Was second song better (b), worse (w), or the same ([ENTER])? ')
                    try:
                        df_rank[r][c] = float(ans)
                    except (ValueError, TypeError):
                        df_rank.idx[r, c] = ans
                    fout.write(pair_scores)
                    scores.append(pair_scores)
except Exception as e:
    print(e)
    df = pd.DataFrame(df_rank, columns=filenames, index=filenames)
    df.to_csv(GEN_DIR / 'pairwise_scores.csv')
