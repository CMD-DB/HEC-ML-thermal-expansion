"""QHA-calibrated CTE transfer and fixed-input ROM, Turner and Kerner comparisons."""
import numpy as np
import pandas as pd
from scipy.integrate import quad

from prepare_data import DATA, OUT, TARGETS
from train_models import CFG, score

QKEY = ['HfNbTaTiV', 'HfMoNbTaTi', 'HfNbTaTiW']
HECS = ['HECV', 'HECMo', 'HECW']
XS = np.linspace(0, 60, 12001)
nodes, weights = np.polynomial.legendre.leggauss(80)
u = XS[:, None] * (nodes + 1) / 2
integrand = np.zeros_like(u)
nz = u > 0
integrand[nz] = u[nz]**4 * np.exp(-u[nz]) / (-np.expm1(-u[nz]))**2
DX = np.ones_like(XS)
DX[1:] = 3 / XS[1:]**3 * XS[1:] / 2 * (integrand[1:] @ weights)


def alpha(theta, bulk, volume, temperature, gamma=1, correction=1):
    theta, bulk, volume, temperature = np.broadcast_arrays(theta, bulk, volume, temperature)
    temp = np.where(temperature > 0, temperature, 1)
    cv = 3 * 1.380649e-23 * np.interp(theta / temp, XS, DX)
    return np.where(temperature > 0,
                    gamma * correction * cv / (3 * bulk * 1e9 * volume * 1e-30) * 1e6, 0)


def draws(mu, sd, rng):
    v = rng.normal(np.asarray(mu)[:, None], np.asarray(sd)[:, None],
                   (len(mu), CFG['monte_carlo_samples']))
    bad = v <= 0
    while bad.any():
        v[bad] = rng.normal(np.broadcast_to(np.asarray(mu)[:, None], v.shape)[bad],
                            np.broadcast_to(np.asarray(sd)[:, None], v.shape)[bad])
        bad = v <= 0
    return v


def mix(method, ap, am, fm, kp, km, gm):
    base = (1 - fm) * ap + fm * am
    if method == 'ROM':
        return base
    if method == 'Turner':
        return (fm * km * am + (1 - fm) * kp * ap) / (fm * km + (1 - fm) * kp)
    if method == 'Kerner':
        den = fm * km + (1 - fm) * kp + 3 * km * kp / (4 * gm)
        return base + fm * (1 - fm) * (ap - am) * (kp - km) / den
    raise ValueError(method)


def metric(q):
    y, p = q.FEM.to_numpy(), q.predicted.to_numpy()
    curves = q.assign(abs_error=abs(p-y), ape=100*abs(p-y)/abs(y))
    curves = curves.groupby(['hec', 'metal', 'Vm']).agg(MAE=('abs_error', 'mean'), MAPE=('ape', 'mean'))
    return dict(n_curves=len(curves), n_correlated_points=len(q),
                **{k: v for k, v in score(y, p).items() if k != 'n'},
                MAPE_percent=float(np.mean(100*abs(p-y)/abs(y))),
                mean_curve_MAPE_percent=float(curves.MAPE.mean()), max_curve_MAE=float(curves.MAE.max()))


def main():
    rng = np.random.default_rng(CFG['seed'] + 1)
    d = pd.read_csv(OUT / 'hec_unique.csv')
    p = pd.read_csv(OUT / 'property_predictions.csv')
    fem = pd.read_csv(DATA / 'thermal_curves.csv')
    t = np.arange(0, 3001, 100, dtype=float)
    for _, g in fem.groupby(['hec', 'T']):
        assert g.a_p.nunique() == 1
    qha = fem.drop_duplicates(['hec', 'T']).pivot(index='T', columns='hec', values='a_p')
    a = qha.loc[t, HECS].to_numpy().T
    copied = (fem.hec == 'HECV') & (fem.metal == 'Nb')
    anomalous = (fem.hec == 'HECW') & (fem.metal == 'Nb') & (fem.Vm == 10)
    fem['valid'] = ~(copied | anomalous)
    fem['exclusion_reason'] = np.where(copied, 'copied Ta panel',
                                      np.where(anomalous, 'anomalous original curve', ''))
    exclusions = fem.groupby(['hec', 'metal', 'Vm']).agg(
        valid=('valid', 'first'), reason=('exclusion_reason', 'first'), source=('source', 'first'))
    assert len(exclusions) == 60 and exclusions.valid.sum() == 54
    exclusions.to_csv(OUT / 'fem_curve_exclusions.csv')
    nb = fem[copied].sort_values(['Vm', 'T']).a_c.to_numpy()
    ta = fem[(fem.hec == 'HECV') & (fem.metal == 'Ta')].sort_values(['Vm', 'T']).a_c.to_numpy()
    assert np.array_equal(nb, ta)
    folds = pd.read_csv(OUT / 'fold_membership.csv')
    train_keys = set(folds.loc[(folds.fold == 'TiTaHfNb') & (folds.role == 'train'), 'key'])
    assert not train_keys.intersection(QKEY)
    mu, sd = {}, {}
    for target in TARGETS:
        q = p[p.target == target]
        mean = q.groupby('key').predicted.mean()
        var = q.assign(second=q.sigma**2 + q.predicted**2).groupby('key').second.mean() - mean**2
        mu[target] = mean.reindex(d.key).to_numpy()
        sd[target] = np.sqrt(var.reindex(d.key).to_numpy())
    indices = {key: i for i, key in enumerate(d.key)}
    base = np.array([alpha(d.thetaD.iloc[indices[k]], d.B.iloc[indices[k]],
                           d.V0.iloc[indices[k]], t) for k in QKEY])
    fitmask = (t >= 300) & (t <= 2000)
    gamma = np.array([np.dot(b[fitmask], q[fitmask]) / np.dot(b[fitmask], b[fitmask])
                      for q, b in zip(a, base)])
    factors = np.ones_like(a)
    factors[:, 1:] = a[:, 1:] / (gamma[:, None] * base[:, 1:])
    for ratio in [.05, .2, .35, 1, 3, 7, 12, 30]:
        exact = 3 / ratio**3 * quad(lambda x: x**4*np.exp(-x)/(-np.expm1(-x))**2,
                                    0, ratio, epsabs=1e-12)[0]
        assert abs(np.interp(ratio, XS, DX) - exact) < 1e-7
    # Retain the original random stream after its 136-composition uncertainty draws.
    for target in TARGETS:
        draws(mu[target], sd[target], rng)
    qrows, qmetrics = [], []
    for i, (key, hec) in enumerate(zip(QKEY, HECS)):
        j = indices[key]
        assert d.families.iloc[j] == 'TiTaHfNb'
        others = [z for z in range(3) if z != i]
        gh, fh = gamma[others].mean(), factors[others].mean(axis=0)
        theta, bulk, volume = [draws([mu[target][j]], [sd[target][j]], rng)[0] for target in TARGETS]
        samples = alpha(theta[:, None], bulk[:, None], volume[:, None], t[None, :], gh, fh).T
        mid = np.median(samples, axis=-1)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=-1)
        physics = alpha(d.thetaD.iloc[j], d.B.iloc[j], d.V0.iloc[j], t, gh, fh)
        donors = '|'.join(QKEY[z] for z in others)
        assert key not in donors.split('|')
        for temp, ref, pred, low, high, phys in zip(t, a[i], mid, lo, hi, physics):
            qrows.append(dict(hec=hec, key=key, T=temp, QHA=ref, predicted=pred,
                              lo=low, hi=high, physics_only=phys, calibration_keys=donors))
        for maxt in [2000, 3000]:
            mask = (t >= 300) & (t <= maxt)
            qmetrics.append(dict(hec=hec, Tmin=300, Tmax=maxt, n_temperatures=int(mask.sum()),
                                 MAE=np.mean(abs(mid[mask]-a[i, mask])),
                                 MAPE_percent=100*np.mean(abs(mid[mask]-a[i, mask])/a[i, mask]),
                                 coverage95=np.mean((a[i, mask] >= lo[mask]) & (a[i, mask] <= hi[mask])),
                                 physics_only_MAPE_percent=100*np.mean(abs(physics[mask]-a[i, mask])/a[i, mask])))
    qr, qm = pd.DataFrame(qrows), pd.DataFrame(qmetrics)
    qr.to_csv(OUT / 'qha_holdout_curves.csv', index=False)
    qm.to_csv(OUT / 'qha_holdout_metrics.csv', index=False)
    metals = pd.read_csv(DATA / 'metal_elastic.csv')
    metals['K_GPa'] = metals.E_GPa / (3 * (1 - 2 * metals.nu))
    metals['G_GPa'] = metals.E_GPa / (2 * (1 + metals.nu))
    bulk_predictions = p[(p.fold == 'TiTaHfNb') & (p.target == 'B')]
    mapping = qr[['hec', 'key']].drop_duplicates().merge(d[['key', 'B']], on='key', validate='one_to_one')
    mapping = mapping.merge(bulk_predictions[['key', 'predicted']].rename(columns={'predicted': 'B_ML'}),
                            on='key', validate='one_to_one')
    z = fem[fem.valid].rename(columns={'a_c': 'FEM'}).merge(
        qr[['hec', 'T', 'QHA', 'predicted']], on=['hec', 'T'], validate='many_to_one')
    z = z.rename(columns={'predicted': 'ML_HEC'}).merge(mapping, on='hec', validate='many_to_one')
    z = z.merge(metals, on='metal', validate='many_to_one')
    z['f_m'] = z.Vm / 100
    melting = pd.read_csv(DATA / 'elemental_properties.csv').set_index('element').MT
    z['solid_state_valid'] = z['T'] < z.metal.map(melting)
    assert np.allclose(z.a_p, z.QHA)
    rows = []
    for mode, ap, kp in [('QHA_DFT_B', z.QHA, z.B), ('ML_CTE_DFT_B', z.ML_HEC, z.B),
                         ('ML_CTE_ML_B', z.ML_HEC, z.B_ML)]:
        for method in ['ROM', 'Turner', 'Kerner']:
            g = z.copy()
            g['mode'], g['method'] = mode, method
            g['predicted'] = mix(method, ap, z.a_m, z.f_m, kp, z.K_GPa, z.G_GPa)
            rows.append(g)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUT / 'composite_predictions.csv', index=False)
    metrics, groups = [], []
    for (mode, method), g in result.groupby(['mode', 'method']):
        for scope in ['300_2000', '300_3000_solid']:
            mask = g['T'].between(300, 2000 if scope == '300_2000' else 3000)
            if scope == '300_3000_solid':
                mask &= g.solid_state_valid
            q = g[mask]
            metrics.append(dict(mode=mode, method=method, scope=scope, **metric(q)))
            for fields in [['metal'], ['hec'], ['hec', 'metal', 'Vm']]:
                for keys, subgroup in q.groupby(fields):
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    groups.append(dict(mode=mode, method=method, scope=scope, grouping='/'.join(fields),
                                       **dict(zip(fields, keys)), **metric(subgroup)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(OUT / 'composite_metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(OUT / 'composite_group_metrics.csv', index=False)
    main_metrics = metrics[(metrics['mode'] == 'ML_CTE_ML_B') & (metrics.scope == '300_3000_solid')]
    assert main_metrics.n_curves.eq(54).all() and main_metrics.n_correlated_points.eq(1455).all()
    print('Mean QHA MAPE [%]:', qm.loc[qm.Tmax == 3000, 'MAPE_percent'].mean())
    print(main_metrics.to_string(index=False))


if __name__ == '__main__':
    main()
