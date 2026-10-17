"""
revision_analyses.py -- drop-in analyses for the reviewer comments.
None of these retrain the CNNs; most only need the val/test scores that
run_final_pipeline.py already computes.

Typical use inside run_hybrid_system(), after ensemble_*_probs exist:

    from revision_analyses import *
    protocol_report(train_pairs, val_pairs, test_pairs)                    # Comment 2
    mp_val  = per_model_probs(val_loader,  model_paths, DeeperCNN, DEVICE)
    mp_test = per_model_probs(test_loader, model_paths, DeeperCNN, DEVICE)
    ensemble_size_ablation(y_test, mp_test, "test")                        # Comment 3
    alpha_sweep(y_val, ensemble_val_probs, xgb_val_probs,
                y_test, ensemble_test_probs, xgb_test_probs)               # Comment 3
    xgb_feature_ablation(X_train, y_train, X_val, y_val, X_test, y_test)   # Comment 3
    val_selected_operating_points(y_val, final_val_probs,
                                  y_test, final_probs)                     # Comment 6
    u1, u2 = pair_uids(test_pairs)
    subject_bootstrap(y_test, {"hybrid": final_probs, "cnn": ensemble_test_probs}, u1, u2)  # Comment 4
"""
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_curve, auc


# ----------------------------------------------------------------- helpers
def _uid(graph):
    # graph_id = "{uid}_{hand}_{finger}_{impression}"
    return graph.graph_id.split("_")[0]


def pair_uids(pairs):
    return (np.array([_uid(p[0]) for p in pairs]),
            np.array([_uid(p[1]) for p in pairs]))


def _eer(fpr, tpr):
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def _auc_eer(y, s):
    f, t, _ = roc_curve(y, s)
    return auc(f, t), _eer(f, t)


# ------------------------------------------- 1. protocol audit (Comment 2)
def protocol_report(train_pairs, val_pairs, test_pairs):
    """Per-split pair counts, impostor:genuine ratio, duplicates, and an
    assertion that no subject or impression is shared across splits."""
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    users, graphs = {}, {}
    print(f"{'split':<6}{'genuine':>9}{'impostor':>10}{'imp:gen':>9}"
          f"{'subjects':>10}{'images':>8}{'dup pairs':>10}")
    for name, pairs in splits.items():
        labels = np.array([p[2] for p in pairs])
        n_g, n_i = int((labels == 1).sum()), int((labels == 0).sum())
        users[name] = {_uid(p[0]) for p in pairs} | {_uid(p[1]) for p in pairs}
        graphs[name] = {p[0].graph_id for p in pairs} | {p[1].graph_id for p in pairs}
        keys = [frozenset((p[0].graph_id, p[1].graph_id)) for p in pairs]
        dups = len(keys) - len(set(keys))
        print(f"{name:<6}{n_g:>9}{n_i:>10}{n_i / max(n_g, 1):>9.2f}"
              f"{len(users[name]):>10}{len(graphs[name]):>8}{dups:>10}")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (users[a] & users[b]), f"subject leakage {a}/{b}"
        assert not (graphs[a] & graphs[b]), f"impression leakage {a}/{b}"
    print("OK: no subject or impression shared across splits.")


# ---------------------------------------- 2. alpha sensitivity (Comment 3)
def alpha_sweep(y_val, cnn_val, xgb_val, y_test, cnn_test, xgb_test,
                alphas=np.linspace(0, 1, 101), save_path="alpha_sensitivity.png"):
    rows = []
    for a in alphas:
        av, _ = _auc_eer(y_val, a * cnn_val + (1 - a) * xgb_val)
        at, et = _auc_eer(y_test, a * cnn_test + (1 - a) * xgb_test)
        rows.append((a, av, at, et))
    rows = np.array(rows)
    best = rows[np.argmax(rows[:, 1])]
    plateau = rows[rows[:, 1] >= rows[:, 1].max() - 0.002, 0]
    print(f"best val alpha={best[0]:.2f} (val AUC {best[1]:.4f}, test AUC {best[2]:.4f})")
    print(f"alpha range within 0.002 val-AUC of the optimum: "
          f"[{plateau.min():.2f}, {plateau.max():.2f}]")
    print(f"endpoints: alpha=0 (XGB) test AUC {rows[0, 2]:.4f}, "
          f"alpha=1 (CNN) test AUC {rows[-1, 2]:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(rows[:, 0], rows[:, 1], label="validation")
    ax[0].plot(rows[:, 0], rows[:, 2], label="test")
    ax[0].axvline(best[0], color="k", ls=":", lw=1)
    ax[0].set_xlabel(r"fusion weight $\alpha$ (CNN)"); ax[0].set_ylabel("AUC"); ax[0].legend()
    ax[1].plot(rows[:, 0], rows[:, 3], color="C2")
    ax[1].axvline(best[0], color="k", ls=":", lw=1)
    ax[1].set_xlabel(r"fusion weight $\alpha$ (CNN)"); ax[1].set_ylabel("test EER")
    for a_ in ax: a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=300)
    return rows


# --------------------------------------- 3. CNN ensemble size (Comment 3)
def per_model_probs(loader, model_paths, model_cls, device):
    """Returns array [n_models, n_pairs] of sigmoid scores."""
    import torch
    out = []
    for p in model_paths:
        m = model_cls().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        probs = []
        with torch.no_grad():
            for a, b, _ in loader:
                logits = m(a.to(device), b.to(device))
                probs.extend(np.atleast_1d(torch.sigmoid(logits).cpu().numpy()))
        out.append(np.array(probs))
        del m
    return np.stack(out)


def ensemble_size_ablation(y, model_probs, tag=""):
    M = model_probs.shape[0]
    for k in range(1, M + 1):
        res = np.array([_auc_eer(y, model_probs[list(c)].mean(0))
                        for c in combinations(range(M), k)])
        print(f"[{tag}] {k} model(s): AUC {res[:, 0].mean():.4f} "
              f"(range {res[:, 0].min():.4f}-{res[:, 0].max():.4f}) | "
              f"EER {res[:, 1].mean():.4f}")


# ------------------------------- 4. XGBoost feature ablation (Comment 3)
BLOCK = 30  # per-graph summary vector length; pair vector = [|diff|, chi2, prod]
GROUPS = {
    "spatial_mean": range(0, 5), "spatial_std": range(5, 10),
    "edge_dist_hist": range(10, 17), "edge_dist_median": [17],
    "angle_hist": range(18, 25), "bif_ratio": [25], "local_density": range(26, 30),
}
OPERATORS = {"abs_diff": 0, "chi_sq": 1, "product": 2}


def _cols(drop_groups=(), ops=(0, 1, 2)):
    return [o * BLOCK + c for o in ops for g, r in GROUPS.items()
            if g not in drop_groups for c in r]


def dead_features(X_train):
    """Columns with zero variance on the training set (e.g. empty histogram bins)."""
    dead = np.where(np.asarray(X_train).std(0) < 1e-9)[0]
    print(f"{len(dead)} constant feature columns: {dead.tolist()}")
    return dead


def xgb_feature_ablation(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    import xgboost as xgb
    configs = {"full (90-d)": _cols()}
    for name, o in OPERATORS.items():
        configs[f"only {name}"] = _cols(ops=(o,))
    for g in GROUPS:
        configs[f"without {g}"] = _cols(drop_groups=(g,))
    results = {}
    for name, cols in configs.items():
        m = xgb.XGBClassifier(n_estimators=2000, max_depth=6, learning_rate=0.02,
                              eval_metric="logloss", early_stopping_rounds=60,
                              n_jobs=-1, random_state=seed)
        m.fit(X_train[:, cols], y_train, eval_set=[(X_val[:, cols], y_val)], verbose=False)
        a, e = _auc_eer(y_test, m.predict_proba(X_test[:, cols])[:, 1])
        results[name] = (a, e)
        print(f"{name:<28} dims={len(cols):>3}  test AUC {a:.4f}  EER {e:.4f}")
    return results


# ------------------------- 5. val-selected thresholds (Comment 6)
def threshold_at_far(y_val, s_val, far):
    fpr, _, thr = roc_curve(y_val, s_val)
    ok = np.where(fpr <= far)[0]
    return thr[ok[-1]]


def val_selected_operating_points(y_val, s_val, y_test, s_test,
                                  fars=(0.1, 0.01, 0.001)):
    """Pick the threshold on VALIDATION for a target FAR, then report the
    realised FAR/TAR on TEST. Also prints the FAR resolution of the test set."""
    y_val, y_test = np.asarray(y_val), np.asarray(y_test)
    s_test = np.asarray(s_test)
    n_imp = int((y_test == 0).sum())
    print(f"test impostor pairs: {n_imp} -> FAR resolution = {1 / n_imp:.4%} per pair")
    for far in fars:
        t = threshold_at_far(y_val, s_val, far)
        acc = s_test >= t
        far_t, tar_t = acc[y_test == 0].mean(), acc[y_test == 1].mean()
        print(f"target FAR {far:.2%}: thr={t:.4f} | test FAR {far_t:.3%} "
              f"TAR {tar_t:.4f} FRR {1 - tar_t:.4f} "
              f"(~{far * n_imp:.1f} impostor pairs expected above threshold)")


# ------------------- 6. subject-level (cluster) bootstrap (Comment 4)
def _prep(uid1, uid2):
    subjects = np.unique(np.concatenate([uid1, uid2]))
    idx = {s: i for i, s in enumerate(subjects)}
    return (np.array([idx[u] for u in uid1]),
            np.array([idx[u] for u in uid2]), len(subjects))


def _wmetric(y, s, w, metric):
    m = w > 0
    y, s, w = y[m], s[m], w[m]
    if len(np.unique(y)) < 2:
        return np.nan
    f, t, _ = roc_curve(y, s, sample_weight=w)
    return auc(f, t) if metric == "auc" else _eer(f, t)


def subject_bootstrap(y, scores, uid1, uid2, metric="auc", n_boot=2000,
                      seed=42, ci=95):
    """Resamples SUBJECTS with replacement (not pairs). A genuine pair of a
    subject drawn k times counts k times; an impostor pair (a,b) counts
    k_a*k_b times. scores: dict name -> score array. If >=2 systems, also
    returns the paired difference (first - second) and a one-sided p."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    i1, i2, S = _prep(uid1, uid2)
    same = i1 == i2
    names = list(scores)
    draws = {n: [] for n in names}
    diffs = []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, S, S), minlength=S)
        w = np.where(same, counts[i1], counts[i1] * counts[i2]).astype(float)
        vals = {n: _wmetric(y, np.asarray(scores[n]), w, metric) for n in names}
        if any(np.isnan(v) for v in vals.values()):
            continue
        for n in names:
            draws[n].append(vals[n])
        if len(names) >= 2:
            diffs.append(vals[names[0]] - vals[names[1]])
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    out = {n: (float(np.mean(d)), float(np.percentile(d, lo)), float(np.percentile(d, hi)))
           for n, d in draws.items()}
    for n, (m, l, h) in out.items():
        print(f"{n}: {metric.upper()} {m:.4f}  {ci}% CI [{l:.4f}, {h:.4f}]  (subject-level)")
    if diffs:
        d = np.array(diffs)
        out["diff"] = (float(d.mean()), float(np.percentile(d, lo)), float(np.percentile(d, hi)),
                       float(np.mean(d <= 0)))
        print(f"{names[0]} - {names[1]}: {d.mean():+.4f}  CI [{out['diff'][1]:+.4f}, "
              f"{out['diff'][2]:+.4f}]  one-sided p (diff<=0) = {out['diff'][3]:.4f}")
    return out