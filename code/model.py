"""Ember Guard baseline model: explainable logistic regression that ESTIMATES the probability
that a flame could be sustained (risk) given O2 %, airflow and fuel type. It does NOT detect fires.
Validation: GroupKFold by test_id (all rows of a test are held out together, never seen in training),
repeated 20x with shuffled groups; compared with a no-skill baseline.
Usage: python -I model.py ../data/bass2_dataset.csv ../model/"""
import sys, os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
df = pd.read_csv(sys.argv[1]); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
df = df[df.material.str.startswith("PMMA") | (df.material == "SIBAL fabric")].copy()
def family(r):
    if r.material == "SIBAL fabric": return "fabric"
    if r.geometry == "rod": return "rod"
    return "thin_film" if r.thickness_mm < 0.5 else "thick_slab"
df["family"] = df.apply(family, axis=1)
df["y"] = (df.outcome == "burning").astype(int)
df["lf"] = np.log10(df.flow_cm_s + 0.1)
FAM = ["thin_film", "thick_slab", "rod", "fabric"]
def feats(d):
    X = pd.DataFrame({"o2_pct": d.o2_pct, "log_flow": d.lf, "log_flow_sq": d.lf ** 2})
    for f in FAM[1:]: X["fam_" + f] = (d.family == f).astype(int)
    return X
X, y, g = feats(df), df.y.values, df.test_id.values
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
aucs, baccs, briers = [], [], []
rng = np.random.default_rng(0); ug = np.unique(g)
for rep in range(20):
    perm = dict(zip(ug, rng.permutation(len(ug))))
    gg = np.array([perm[t] for t in g]); p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gg):
        p[te] = mk().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    aucs.append(roc_auc_score(y, p)); baccs.append(balanced_accuracy_score(y, p > .5)); briers.append(brier_score_loss(y, p))
base_brier = brier_score_loss(y, np.full(len(y), y.mean()))
m = mk().fit(X, y); sc, lr = m[0], m[1]
coef = lr.coef_[0] / sc.scale_; inter = lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_)
# bootstrap by test for uncertainty band ("margen de error")
boot = []
for b in range(300):
    pick = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(g == t)[0] for t in pick])
    if len(set(y[idx])) < 2: continue
    mb = mk().fit(X.iloc[idx], y[idx]); s, l = mb[0], mb[1]
    cb = l.coef_[0] / s.scale_; ib = l.intercept_[0] - np.sum(l.coef_[0] * s.mean_ / s.scale_)
    boot.append([ib] + list(cb))
res = {
  "what": "Estimated probability that a flame can be sustained (fire risk), NOT a fire detector",
  "n_rows": int(len(df)), "n_tests": int(len(ug)),
  "features": list(X.columns), "intercept": float(inter), "coef": dict(zip(X.columns, map(float, coef))),
  "formula": "p = 1/(1+exp(-(intercept + sum coef_i*x_i))), log_flow = log10(flow_cm_s + 0.1)",
  "bootstrap_params": boot,
  "validation": {"method": "GroupKFold(5) by test, 20 repeats; held-out tests never seen in training",
     "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
     "balanced_accuracy_mean": float(np.mean(baccs)), "brier_mean": float(np.mean(briers)),
     "brier_no_skill": float(base_brier)},
  "valid_ranges": {"o2_pct": [float(df.o2_pct.min()), float(df.o2_pct.max())], "flow_cm_s": [0.4, 47], "pressure_atm": 1},
  "source": "NASA BASS-II, Olson et al. 2021, NASA/TM-20210011385, Appendix A"}
json.dump(res, open(os.path.join(outdir, "ember_guard_model.json"), "w"), indent=1)
v = res["validation"]
print(f"rows={res['n_rows']} tests={res['n_tests']} burning={y.sum()} extinction={len(y)-y.sum()}")
print(f"held-out AUC {v['auc_mean']:.3f}±{v['auc_sd']:.3f} | balanced acc {v['balanced_accuracy_mean']:.3f} | Brier {v['brier_mean']:.3f} (no-skill {v['brier_no_skill']:.3f})")
print("coefficients (raw units):"); [print(f"  {k:14s} {c:+.3f}") for k, c in res["coef"].items()]; print(f"  intercept      {inter:+.3f}")
B = np.array(boot)
def pr(o2, fl, fam):
    x = np.array([o2, np.log10(fl+.1), np.log10(fl+.1)**2] + [int(fam == f) for f in FAM[1:]])
    ps = 1/(1+np.exp(-(B[:,0] + B[:,1:] @ x))); p0 = 1/(1+np.exp(-(inter + coef @ x)))
    return p0, np.percentile(ps, 5), np.percentile(ps, 95)
print("examples, P(sustained flame) [90% band]:")
for o2, fl, fam in [(21,0.3,"thin_film"),(21,5,"thin_film"),(21,5,"thick_slab"),(17,0.5,"rod"),(17,5,"rod"),(15,5,"rod"),(14,5,"rod"),(17,40,"fabric")]:
    p0, lo, hi = pr(o2, fl, fam); print(f"  O2 {o2}%, flow {fl} cm/s, {fam:10s}: {p0:.2f} [{lo:.2f}-{hi:.2f}]")
