"""Nested family-excluded SVR validation and training-only uncertainty estimation."""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from prepare_data import ROOT, OUT, FEATURES, TARGETS

CFG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def splits(d):
    for family in sorted(set('|'.join(d.families).split('|'))):
        mask = d.families.str.split('|').map(lambda groups: family in groups).to_numpy()
        yield family, np.flatnonzero(~mask), np.flatnonzero(mask)


def tune(x, y, d):
    cv = [(tr, te) for _, tr, te in splits(d)]
    assert min(len(tr) for tr, te in cv) > 0
    return GridSearchCV(make_pipeline(StandardScaler(), SVR()), CFG['svr_grid'],
                        scoring='neg_mean_absolute_error', cv=cv,
                        n_jobs=CFG['jobs'], error_score='raise').fit(x, y)


def score(y, p):
    return dict(n=len(y), R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))


def train():
    d = pd.read_csv(OUT / 'hec_unique.csv')
    x = d[FEATURES].to_numpy()
    rng = np.random.default_rng(CFG['seed'])
    rows, params, folds = [], [], []
    for family, tr, te in splits(d):
        dt = d.iloc[tr].reset_index(drop=True)
        assert set(d.key.iloc[tr]).isdisjoint(d.key.iloc[te])
        folds.extend(dict(fold=family, role=role, key=d.key.iloc[i], families=d.families.iloc[i])
                     for role, indices in [('train', tr), ('test', te)] for i in indices)
        for target in TARGETS:
            y = d[target].to_numpy()
            search = tune(x[tr], y[tr], dt)
            model = search.best_estimator_
            residuals = []
            for _, itr, ite in splits(dt):
                inner = tune(x[tr][itr], y[tr][itr], dt.iloc[itr].reset_index(drop=True))
                residuals.extend((y[tr][ite] - inner.predict(x[tr][ite])).tolist())
            noise = float(np.sqrt(np.mean(np.square(residuals))))
            boot = []
            for _ in range(CFG['bootstrap_repeats']):
                indices = rng.integers(0, len(tr), len(tr))
                fit = clone(model).fit(x[tr][indices], y[tr][indices])
                boot.append(fit.predict(x[te]))
            pred = model.predict(x[te])
            bsd = np.std(boot, axis=0, ddof=1)
            sd = np.sqrt(bsd**2 + noise**2)
            params.append(dict(fold=family, target=target, inner_MAE=-search.best_score_,
                               noise_RMS=noise, **search.best_params_))
            for i, mu, bs, s in zip(te, pred, bsd, sd):
                rows.append(dict(fold=family, key=d.key.iloc[i], target=target,
                                 observed=y[i], predicted=mu, bootstrap_sd=bs,
                                 residual_rms=noise, sigma=s, lo=mu-1.96*s, hi=mu+1.96*s))
            print(f'{family} {target}: R2={score(y[te], pred)["R2"]:.3f}', flush=True)
    predictions = pd.DataFrame(rows)
    metrics, fold_metrics = [], []
    for target, q in predictions.groupby('target'):
        averaged = q.groupby('key')[['observed', 'predicted']].mean()
        metrics.append(dict(target=target, **score(averaged.observed, averaged.predicted)))
        for family, g in q.groupby('fold'):
            fold_metrics.append(dict(target=target, fold=family,
                                     **score(g.observed, g.predicted),
                                     coverage95=float(((g.observed >= g.lo) & (g.observed <= g.hi)).mean())))
    predictions.to_csv(OUT / 'property_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT / 'property_metrics.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT / 'property_fold_metrics.csv', index=False)
    pd.DataFrame(params).to_csv(OUT / 'selected_hyperparameters.csv', index=False)
    pd.DataFrame(folds).to_csv(OUT / 'fold_membership.csv', index=False)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    train()
