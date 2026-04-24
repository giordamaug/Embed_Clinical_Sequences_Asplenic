import pandas as pd
import itertools
from scipy.stats import friedmanchisquare, wilcoxon
import statsmodels.stats.multitest as smm
import numpy as np

# ======================
# FRIEDMAN TEST
# ======================
def friedman_test(df, alpha = 0.05):
    stat, p = friedmanchisquare(*[df[col] for col in df.columns])

    print("\n=== FRIEDMAN TEST ===")
    print(f"Statistic: {stat:.4f}")
    print(f"p-value: {p:.6f}")

    if p < alpha:
        print("➡️ SIGNIFICANT global differences")
    else:
        print("➡️ No global differences")
    return stat, p

# ========================
# WILCOXON POST-HOC + HOLM
# ========================
def wilcoxon_test(df, stat, alpha = 0.05):
    pairs = list(itertools.combinations(df.columns, 2))

    wilcoxon_raw_p = []
    wilcoxon_results = []

    for a, b in pairs:
        try:
            stawinst, pval = wilcoxon(df[a], df[b])
        except ValueError:
            stat, pval = None, 1.0
        wilcoxon_raw_p.append(pval)
        wilcoxon_results.append((a, b, stat, pval))

    wilcoxon_reject, wilcoxon_p_corr, _, _ = smm.multipletests(
        wilcoxon_raw_p, alpha=alpha, method="holm"
    )

    print("\n" + "="*60)
    print("WILCOXON PAIRWISE POST-HOC + HOLM")
    print("="*60)

    found_sig = False
    for i, (a, b, stat, pval) in enumerate(wilcoxon_results):
        if wilcoxon_reject[i]:
            found_sig = True
            print(f"✅ {a} vs {b} | raw p = {pval:.6f} | Holm p = {wilcoxon_p_corr[i]:.6f}")

    if not found_sig:
        print("wins❌ no significant pair after Holm's correction")

    print("\nDetailed Wilcoxon report:")
    for i, (a, b, stat, pval) in enumerate(wilcoxon_results):
        mark = "✅" if wilcoxon_reject[i] else "  "
        print(f"{mark} {a:>8s} vs {b:<8s} | raw p = {pval:.6f} | Holm p = {wilcoxon_p_corr[i]:.6f}")

    methods = list(df.columns)
    n = len(methods)

    # correct p-value matrix
    p_matrix = np.ones((n, n))

    # fill-in the matrix
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            p_matrix[i, j] = wilcoxon_p_corr[k]
            p_matrix[j, i] = wilcoxon_p_corr[k]
            k += 1
    return pd.DataFrame(p_matrix, columns=methods, index=methods), wilcoxon_reject, wilcoxon_p_corr

# ========================
# WILCOXON RANKING
# ========================

def wilcoxon_ranking(df, wilcoxon_reject):
    methods = list(df.columns)
    n = len(methods)

    # inizializza punteggi
    wins = {m: 0 for m in methods}
    losses = {m: 0 for m in methods}

    k = 0
    for i in range(n):
        for j in range(i+1, n):
            a = methods[i]
            b = methods[j]
            
            if wilcoxon_reject[k]:  # significativo
                # chi ha media maggiore?
                if df[a].mean() > df[b].mean():
                    wins[a] += 1
                    losses[b] += 1
                else:
                    wins[b] += 1
                    losses[a] += 1
            k += 1

    # crea dataframe ranking
    ranking_df = pd.DataFrame({
        "wins": wins,
        "losses": losses
    })
    ranking_df["score"] = ranking_df["wins"] - ranking_df["losses"]
    return ranking_df.sort_values(by="score", ascending=False)