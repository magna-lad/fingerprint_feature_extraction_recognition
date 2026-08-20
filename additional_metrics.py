import numpy as np
from sklearn.metrics import roc_curve, auc


def tar_at_far(y_true, y_scores, far_targets=(0.1, 0.01, 0.001)):
    """
    True Accept Rate at fixed False Accept Rate operating points.
    far_targets are fractions, e.g. 0.01 -> 1% FAR, 0.001 -> 0.1% FAR.
    This is the standard biometrics reporting convention (NIST FRVT-style)
    and matters more than EER for real deployment discussions, since EER
    describes an arbitrary crossover point, not a realistic operating threshold.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    results = {}
    for target in far_targets:
        # largest fpr <= target
        idx = np.searchsorted(fpr, target, side='right') - 1
        idx = max(idx, 0)
        results[f"TAR@FAR={target*100:.3f}%"] = float(tpr[idx])
    return results


def dprime(y_true, y_scores):
    """
    Decidability index (d'): separation between genuine and impostor score
    distributions normalized by their pooled variance. Complements EER by
    showing how cleanly separated the two distributions are, not just where
    they cross. Standard in biometrics since Daugman's iris recognition work.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    genuine = y_scores[y_true == 1]
    impostor = y_scores[y_true == 0]
    mu_g, mu_i = genuine.mean(), impostor.mean()
    var_g, var_i = genuine.var(), impostor.var()
    denom = np.sqrt(0.5 * (var_g + var_i))
    return float(abs(mu_g - mu_i) / denom) if denom > 0 else float('nan')


def calculate_eer(y_true, y_scores):
    """Same EER calc you already use in run_final_pipeline.py, duplicated
    here so bootstrap_metric_ci can call it without a circular import."""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)


def bootstrap_metric_ci(y_true, y_scores, metric_fn=None, n_bootstrap=1000,
                         ci=95, seed=42):
    """
    Bootstrap confidence interval for a metric (default: AUC) over the test
    pairs. Answers "how much would this number move on a different sample
    of the same population?" -- directly addresses the "is a single point
    estimate meaningful" gap.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)

    if metric_fn is None:
        def metric_fn(yt, ys):
            fpr, tpr, _ = roc_curve(yt, ys)
            return auc(fpr, tpr)

    stats = []
    attempts = 0
    while len(stats) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue  # degenerate resample, skip
        stats.append(metric_fn(yt, ys))

    stats = np.array(stats)
    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)
    return float(stats.mean()), (float(lower), float(upper))


def paired_bootstrap_auc_diff(y_true, scores_a, scores_b, n_bootstrap=2000,
                               seed=42):
    """
    Paired bootstrap test on the SAME resampled test pairs each iteration,
    answering: is AUC(scores_a) - AUC(scores_b) significantly different
    from zero? Use this to check whether your hybrid fusion's AUC gain
    over CNN-only alone is a real effect or could be noise from one split.

    p_value_one_sided is the fraction of bootstrap resamples where scores_a
    did NOT beat scores_b (i.e. lower is stronger evidence a > b).
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = len(y_true)

    diffs = []
    attempts = 0
    while len(diffs) < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr_a, tpr_a, _ = roc_curve(yt, scores_a[idx])
        fpr_b, tpr_b, _ = roc_curve(yt, scores_b[idx])
        diffs.append(auc(fpr_a, tpr_a) - auc(fpr_b, tpr_b))

    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_auc_diff": float(diffs.mean()),
        "95%_CI": (float(ci_low), float(ci_high)),
        "p_value_one_sided": float(np.mean(diffs <= 0)),
    }


def plot_det_curve(results, save_path="det_curve.png"):
    """
    results: dict of {label: (y_true, y_scores)}
    Plots FNMR vs FMR on a log-log scale -- the standard DET curve format
    in biometrics literature. Makes differences at low-FAR operating
    points visible, which ROC curves compress into the corner.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    for label, (y_true, y_scores) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        # avoid log(0) issues at the extremes
        eps = 1e-4
        fpr_plot = np.clip(fpr, eps, 1)
        fnr_plot = np.clip(fnr, eps, 1)
        plt.plot(fpr_plot, fnr_plot, label=label)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.title('DET Curve')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"DET curve saved to {save_path}")