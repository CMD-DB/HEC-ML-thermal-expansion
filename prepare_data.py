"""Consolidate compositions and construct the twelve elemental descriptors."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results'
FEATURES = [p + s for p in ['EA', 'MT', 'AR', 'VEC', 'EN', 'AW'] for s in 'AD']
TARGETS = ['thetaD', 'B', 'V0']


def descriptors(elements, properties):
    values = []
    for name in ['EA', 'MT', 'AR', 'VEC', 'EN', 'AW']:
        v = np.array([properties[e][name] for e in elements])
        values.extend([v.mean(), v.std(ddof=0)])
    return values


def prepare():
    raw = pd.read_csv(DATA / 'hec.csv')
    assert len(raw) == 144 and not raw.comp.duplicated().any()
    raw['elements'] = raw.comp.map(lambda c: re.findall(r'[A-Z][a-z]?', c[1:c.index(')')]))
    assert raw.elements.map(len).eq(5).all()
    raw['key'] = raw.elements.map(lambda e: ''.join(sorted(e)))
    raw['family'] = raw.elements.map(lambda e: ''.join(e[:4]))
    rows, duplicates = [], []
    for key, g in raw.groupby('key', sort=True):
        families = '|'.join(sorted(g.family.unique()))
        rows.append(dict(key=key, composition=g.comp.iloc[0], families=families,
                         n_replicates=len(g), **{t: g[t].mean() for t in TARGETS}))
        if len(g) > 1:
            duplicates.append(dict(key=key, families=families, n=len(g),
                                   **{t + '_range': g[t].max() - g[t].min() for t in TARGETS}))
    d = pd.DataFrame(rows)
    properties = pd.read_csv(DATA / 'elemental_properties.csv', float_precision='round_trip')
    properties = properties.set_index('element').to_dict('index')
    x = pd.DataFrame([descriptors(re.findall(r'[A-Z][a-z]?', k), properties)
                      for k in d.key], columns=FEATURES)
    d = pd.concat([d, x], axis=1)
    assert len(d) == 136 and np.isfinite(d[FEATURES + TARGETS].to_numpy()).all()
    OUT.mkdir(exist_ok=True)
    d.to_csv(OUT / 'hec_unique.csv', index=False)
    pd.DataFrame(duplicates).to_csv(OUT / 'duplicate_spread.csv', index=False)
    print('144 entries -> 136 compositions; 12 descriptors', flush=True)


if __name__ == '__main__':
    prepare()
