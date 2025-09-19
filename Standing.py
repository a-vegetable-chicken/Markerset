import os 
import pandas as pd 
import numpy as np
import argparse
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)
from scipy.stats import chi2_contingency
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from collections import defaultdict
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import adjusted_mutual_info_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        conditions_raw = lines[0].strip().split('\t')
        angle_types_raw = lines[1].strip().split('\t')

        conditions = [c.split('\\')[-1][4:-7] for c in conditions_raw]
        angle_types = [a.split('\\')[-1][6:-6] for a in angle_types_raw]

    df = pd.read_csv(file_path, sep='\t', header=None, skiprows=5, encoding='utf-8').iloc[:,1:]
    return conditions, angle_types, df
  

def build_data_dict(root_path):
    data_dict = {}

    for i in range(1, 17):
        folder = f"S{i:02d}"
        file_path = os.path.join(root_path, folder, "Output", "Standing_Right_Angles.txt")

        if not os.path.exists(file_path):
            print(f"file does not exist：{file_path}")
            continue

        try:
            conditions, angle_types, df = read_file(file_path)
        except Exception as e:
            print(f"{folder} can not be read：{e}")
            continue

        data_dict[folder] = {}

        for col_idx in range(0, df.shape[1], 3):
            try:
                condition  = conditions[col_idx]
                angle_type = angle_types[col_idx]

                #for S07 name correction
                if folder == "S07" and condition.startswith("WW-A-0_S"):
                    condition = "WW-A-0_S"
                

                x_values = df.iloc[:, col_idx    ].fillna(method='ffill').fillna(method='bfill').tolist()
                y_values = df.iloc[:, col_idx + 1].fillna(method='ffill').fillna(method='bfill').tolist()
                z_values = df.iloc[:, col_idx + 2].fillna(method='ffill').fillna(method='bfill').tolist()

            except Exception as e:
                print(f"{folder} index {col_idx} error，skip：{e}")
                continue

            if condition not in data_dict[folder]:
                data_dict[folder][condition] = {}
            if angle_type not in data_dict[folder][condition]:
                data_dict[folder][condition][angle_type] = {}

            data_dict[folder][condition][angle_type]['X'] = x_values
            data_dict[folder][condition][angle_type]['Y'] = y_values
            data_dict[folder][condition][angle_type]['Z'] = z_values

    return data_dict

def prepare_ancova(data_dict,
                   conditions,         
                   cov_condition,      
                   angle_type,
                   axis='Y'):

    rows = []
    cond_values = defaultdict(list)

  
    cov_is_pair = isinstance(cov_condition, (list, tuple))

    for sub in data_dict:
        try:
            if cov_is_pair:  # ΔB = cov1 − cov2
                v1 = data_dict[sub][cov_condition[0]][angle_type][axis]
                v2 = data_dict[sub][cov_condition[1]][angle_type][axis]
                cov_mean = np.nanmean(v1) - np.nanmean(v2)
            else:
                v = data_dict[sub][cov_condition][angle_type][axis]
                cov_mean = np.nanmean(v)
        except KeyError:
            continue

        for cond in conditions:
            try:
                if isinstance(cond, (list, tuple)):
                    v_new = data_dict[sub][cond[0]][angle_type][axis]
                    v_ref = data_dict[sub][cond[1]][angle_type][axis]
                    val = np.nanmean(v_new) - np.nanmean(v_ref)
                    cond_name = f'{cond[0]}–{cond[1]}'
                else:
                    v = data_dict[sub][cond][angle_type][axis]
                    val = np.nanmean(v)
                    cond_name = cond
            except KeyError:
                continue

            rows.append({
                'Subject':  sub,
                'Condition': cond_name,
                'Value':     val,
                'Covariate': cov_mean
            })
            cond_values[cond_name].append(val)

    
    if cond_values:
        print("\n📊 Descriptive statistics per condition:")
        for cond, vals in cond_values.items():
            arr = np.array(vals)
            mean = np.nanmean(arr)
            std  = np.nanstd(arr, ddof=1)  
            n    = np.sum(~np.isnan(arr))
            print(f"  {cond}: mean = {mean:.4f}, std = {std:.4f}, n = {n}")

    return pd.DataFrame(rows)

def run_ancova_posthoc(df_long, alpha=0.05):
    if df_long.empty:
        print("❌ data is empty");  return

    model = ols('Value ~ C(Condition) + Covariate', data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== ANCOVA result ===")
    print(anova_table)

    ss_condition = anova_table.loc['C(Condition)', 'sum_sq']
    ss_error     = anova_table.loc['Residual', 'sum_sq']
    eta_squared  = ss_condition / (ss_condition + ss_error)
    cohen_f      = (eta_squared / (1 - eta_squared)) ** 0.5
    print(f"\n📏 Effect size for main effect (Condition):")
    print(f"  η² = {eta_squared:.4f}")
    print(f"  Cohen’s f = {cohen_f:.4f}  → Interpretation: "
          + ("small" if cohen_f < 0.10 else "medium" if cohen_f < 0.25 else "large"))


    exog_names  = model.model.exog_names
    cond_levels = pd.Index(df_long['Condition'].unique().tolist())
    cov_mean    = float(df_long['Covariate'].mean())

    dummy_levels = [n.split('T.',1)[1].rstrip(']') for n in exog_names if n.startswith('C(Condition)[T.')]
    baseline_candidates = [lv for lv in cond_levels if lv not in set(dummy_levels)]
    baseline = baseline_candidates[0] if baseline_candidates else cond_levels[0]

    def exog_row_for(cond, cov_val):
        row = np.zeros(len(exog_names), dtype=float)
        if 'Intercept' in exog_names:
            row[exog_names.index('Intercept')] = 1.0
        if 'Covariate' in exog_names:
            row[exog_names.index('Covariate')] = cov_val
        if cond != baseline:
            dn = f'C(Condition)[T.{cond}]'
            if dn in exog_names:
                row[exog_names.index(dn)] = 1.0
        return row

    V = model.cov_params().values
    rows = []
    for cond in cond_levels:
        x = exog_row_for(cond, cov_mean)
        emm = float(np.dot(x, model.params.values))
        se  = float(np.sqrt(np.dot(x, np.dot(V, x))))
        rows.append({'Condition': cond, 'EMM': emm, 'SE': se})
    emm_df = pd.DataFrame(rows)
    print("\n=== Estimated Marginal Means (adjusted for covariate) ===")
    print(emm_df)

    #posthoc
    print(f"\n=== Pairwise comparisons on adjusted means (Holm correction, α={alpha}) ===")
    results = []
    for i in range(len(cond_levels)):
        for j in range(i+1, len(cond_levels)):
            c1, c2 = cond_levels[i], cond_levels[j]
            L = exog_row_for(c1, cov_mean) - exog_row_for(c2, cov_mean) 
            L2 = np.asarray(L, dtype=float).reshape(1, -1)               
            t_res = model.t_test(L2)
            diff  = float(np.dot(L, model.params.values))
            results.append({
                'group1': c1, 'group2': c2,
                'diff(EMM)': diff,
                't': float(t_res.statistic),
                'p_raw': float(t_res.pvalue)
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['p_adj'] = multipletests(results_df['p_raw'], method='holm')[1]
        results_df['reject'] = results_df['p_adj'] < alpha
    print(results_df)

def check_homogeneity_of_slopes(df_long, alpha=0.05):

    if df_long.empty:
        print("❌ data is empty");  return False, np.nan, None

    m_full = ols('Value ~ C(Condition)*Covariate', data=df_long).fit()
    anova_full = sm.stats.anova_lm(m_full, typ=2) 
    try:
        p_int = float(anova_full.loc['C(Condition):Covariate', 'PR(>F)'])
    except KeyError:
        print("\n=== Homogeneity test (interaction) ===")
        print(anova_full)
        print("⚠️ No interaction term found (possibly fewer than 2 groups or insufficient data)")
        return False, np.nan, m_full

    print("\n=== Test of regression slope homogeneity (Group × Covariate interaction) ===")
    print(anova_full.loc[['C(Condition)', 'Covariate', 'C(Condition):Covariate', 'Residual']])
    print(f"→ Homogeneity p = {p_int:.4g}  => {'passed' if p_int >= alpha else 'not passed'} (α={alpha})")

    return (p_int >= alpha), p_int, m_full

def simple_effects_pairwise(m_full, df_long, c0=None, alpha=0.05):
    if c0 is None:
        c0 = float(df_long['Covariate'].mean())

    cond_levels = pd.Index(df_long['Condition'].unique().tolist())
    exog_names  = m_full.model.exog_names


    dummy_levels = []
    for name in exog_names:
        if name.startswith('C(Condition)[T.') and ']:Covariate' not in name:
            dummy_levels.append(name.split('T.', 1)[1].rstrip(']'))
    baseline_candidates = [lv for lv in cond_levels if lv not in set(dummy_levels)]
    baseline = baseline_candidates[0] if len(baseline_candidates) else cond_levels[0]

    def exog_row_for(cond, cov_val):
        row = np.zeros(len(exog_names), dtype=float)
        if 'Intercept' in exog_names:
            row[exog_names.index('Intercept')] = 1.0
        if 'Covariate' in exog_names:
            row[exog_names.index('Covariate')] = cov_val
        if cond != baseline:
            dn = f'C(Condition)[T.{cond}]'
            if dn in exog_names:
                row[exog_names.index(dn)] = 1.0
            inter = f'C(Condition)[T.{cond}]:Covariate'
            if inter in exog_names:
                row[exog_names.index(inter)] = cov_val
        return row

    rows = []
    for cond in cond_levels:
        x = exog_row_for(cond, c0)
        mean = float(np.dot(x, m_full.params))
        rows.append({'Condition': cond, 'Covariate_at': c0, 'EMM(c0)': mean})
    emm_df = pd.DataFrame(rows)

    results = []
    for i in range(len(cond_levels)):
        for j in range(i+1, len(cond_levels)):
            c1, c2 = cond_levels[i], cond_levels[j]
            L = exog_row_for(c1, c0) - exog_row_for(c2, c0)
            L2 = np.asarray(L, dtype=float).reshape(1, -1)   
            t_res = m_full.t_test(L2)
            diff  = float(np.dot(L, m_full.params))
            results.append({
                'group1': c1, 'group2': c2,
                'Covariate_at': c0,
                'diff(EMM)': diff,
                't': float(t_res.statistic),
                'p_raw': float(t_res.pvalue)
            })
    pairwise_df = pd.DataFrame(results)
    if not pairwise_df.empty:
        pairwise_df['p_adj'] = multipletests(pairwise_df['p_raw'], method='holm')[1]
        pairwise_df['reject'] = pairwise_df['p_adj'] < alpha

    print(f"\n=== Simple effect (at covariate c0 = {c0:.4g}) ===")
    print(emm_df)
    print("\n=== Pairwise (Holm) on simple effects ===")
    print(pairwise_df)

    return emm_df, pairwise_df


def run_ancova_with_homogeneity(df_long, alpha=0.05):

    passed, p_homo, m_full = check_homogeneity_of_slopes(df_long, alpha=alpha)
    out = {'homogeneity_passed': passed, 'p_interaction': p_homo}

    if passed:
        run_ancova_posthoc(df_long, alpha=alpha)
        out['mode'] = 'ANCOVA_main_effect'
    else:

        simple_effects_pairwise(m_full, df_long, c0=float(df_long['Covariate'].mean()), alpha=alpha)
        out['mode'] = 'simple_effects_at_mean'

    return out



def build_subject_condition_matrix(data_dict, angle_type='HFSHK',
                                   axes=('X','Y','Z'), frame_len=201,
                                   pad_mode='edge', subjects=None, conditions=None):

    if isinstance(axes, str):
        axes = (axes,)
    else:
        axes = tuple(axes)

    def _fix_len(seq, L, mode='edge'):
        seq = list(seq) if seq is not None else []
        if len(seq) == 0: return [0.0]*L
        if len(seq) >= L: return seq[:L]
        pad_val = 0.0 if mode == 'zero' else seq[-1]
        return seq + [pad_val]*(L-len(seq))

    rows, metas = [], []
    subj_list = sorted(data_dict.keys()) if subjects is None else list(subjects)
    for subj in subj_list:
        if subj not in data_dict: 
            continue
        cond_list = sorted(data_dict[subj].keys()) if conditions is None else list(conditions)
        for cond in cond_list:
            try:
                axes_dict = data_dict[subj][cond][angle_type]
            except KeyError:
                continue

            feat = []
            ok = True
            for ax in axes:
                seq = axes_dict.get(ax, None)
                if seq is None:
                    ok = False; break
                feat.extend(_fix_len(seq, frame_len, pad_mode))
            if not ok:
                continue

            rows.append(feat)
            metas.append((subj, cond))

    if not rows:
        raise ValueError("❌ check angle/axes/conditions。")

    X = pd.DataFrame(rows)
    expect_cols = len(axes) * frame_len
    if X.shape[1] != expect_cols:
        raise ValueError(f"get {X.shape[1]} cols, should be {expect_cols} (= len(axes)*frame_len).")

    meta = pd.DataFrame(metas, columns=['Subject','Condition'])
    return X, meta



def center_waveforms_by_subject(X, meta, axes=('X','Y','Z'), frame_len=201, mode='subject_mean'):

    if mode == 'none':
        return X.copy()

    # 统一 axes
    if isinstance(axes, str):
        axes = (axes,)
    else:
        axes = tuple(axes)

    Xc = X.copy()
    for subj, idx in meta.groupby('Subject').groups.items():
        block = Xc.loc[idx]
        for ai in range(len(axes)):
            s, e = ai*frame_len, (ai+1)*frame_len
            mu = block.iloc[:, s:e].mean(0)
            Xc.iloc[idx, s:e] = block.iloc[:, s:e] - mu.values
    return Xc


def standardize_pca_for_waveforms(X, use_row_l2=False, pca_var=0.95, random_state=0):


    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    X_work = X_std
    row_norm = None
    if use_row_l2:
        row_norm = Normalizer(norm='l2')
        X_work = row_norm.fit_transform(X_work)

    pca = None
    X_proc = X_work
    if pca_var:
        pca = PCA(n_components=pca_var, random_state=random_state)
        X_proc = pca.fit_transform(X_work)
    return X_std, X_proc, scaler, row_norm, pca

'''
#PCA

def _plot_cov_ellipse(ax, mean2d, cov2d, color, label=None, chi2_val=5.991):  # 95%: chi2(2)=5.991
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2*np.sqrt(vals*chi2_val)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=mean2d, width=width, height=height, angle=theta,
                  facecolor='none', edgecolor=color, lw=2, label=label)
    ax.add_patch(ell)

def pca_scatter_with_ellipses(X_pca2, labels_for_color, label_names_for_ellipse, title, evr_pair, save_dir=None):
 

    def _plot_cov_ellipse(ax, mean2d, cov2d, color, label=None, chi2_val=5.991):  # 95%: chi2(2)=5.991
        vals, vecs = np.linalg.eigh(cov2d)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        width, height = 2*np.sqrt(vals*chi2_val)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        ell = Ellipse(xy=mean2d, width=width, height=height, angle=theta,
                      facecolor='none', edgecolor=color, lw=2, label=label)
        ax.add_patch(ell)

    fig, ax = plt.subplots(figsize=(6.6,4.8))

    for c in np.unique(labels_for_color):
        idx = (labels_for_color == c)
        ax.scatter(X_pca2[idx,0], X_pca2[idx,1], s=40, edgecolor='k', label=f'Cluster {c}', alpha=0.9)

    uniq_names = np.unique(label_names_for_ellipse)
    cmap = plt.cm.get_cmap('tab10', len(uniq_names))
    for i, name in enumerate(uniq_names):
        idx = (label_names_for_ellipse == name)
        if idx.sum() < 3:
            continue
        m = X_pca2[idx].mean(0)
        cov = np.cov(X_pca2[idx].T)
        _plot_cov_ellipse(ax, m, cov, color=cmap(i), label=name)

    ax.set_xlabel(f'PC1 ({evr_pair[0]*100:.1f}%)' if not np.isnan(evr_pair[0]) else 'PC1')
    ax.set_ylabel(f'PC2 ({evr_pair[1]*100:.1f}%)' if not np.isnan(evr_pair[1]) else 'PC2')
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()


    outdir = save_dir or os.path.join(os.getcwd(), "kmeans_figs")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{title.replace(' ', '_').replace('|','_')}_pca.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[saved] {out}")
    plt.close(fig)
'''
def pca_scatter_with_ellipses(X_pca2, labels_for_color, label_names_for_ellipse, title, evr_pair, save_dir=None):

    import os, numpy as np, matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6,4.8))
    ax.tick_params(axis='both', which='major', labelsize=10)   
    ax.tick_params(axis='both', which='minor', labelsize=10)   
    clusters = np.unique(labels_for_color)
    for c in clusters:
        idx = (labels_for_color == c)
        d = c+1
        ax.scatter(X_pca2[idx,0], X_pca2[idx,1], s=42, edgecolor='k', linewidth=0.5,color=colors[c % len(colors)], label=f'Cluster {d}', alpha=0.9)

    ax.set_xlabel(f'Principal component 1',fontsize = 12 )
    ax.set_ylabel(f' Principal component 2',fontsize = 12 )
    ax.set_title(title)
    ax.legend(ncol=1, fontsize=10)
    plt.tight_layout()

    outdir = save_dir or os.path.join(os.getcwd(), "kmeans_figs")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{title.replace(' ', '_').replace('|','_')}_pca.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[saved] {out}")
    plt.close(fig)

def plot_cluster_profiles(X_centered, labels, axes=('X','Y','Z'), frame_len=201, fs=200, title=''):
    import os, numpy as np, matplotlib.pyplot as plt

    if isinstance(axes, str):
        axes = (axes,)
    else:
        axes = tuple(axes)

    outdir = os.path.join(os.getcwd(), "kmeans_figs")

    L = frame_len; t = np.arange(L)/fs
    Ks = np.unique(labels)
    fig, axs = plt.subplots(len(axes), 1, figsize=(7.2, 2.2*len(axes)), sharex=True)
    axs = axs if isinstance(axs, np.ndarray) else [axs]
    for j, axname in enumerate(axes):
        s, e = j*L, (j+1)*L
        for k in Ks:
            idx = (labels==k)
            m = X_centered.iloc[idx, s:e].mean(0).values
            sd = X_centered.iloc[idx, s:e].std(0).values
            axs[j].plot(t, m, label=f'C{k}')
            axs[j].fill_between(t, m-sd, m+sd, alpha=0.15)
        axs[j].set_ylabel(axname)
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend(ncol=len(Ks))
    plt.suptitle(title); plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{title.replace(' ', '_').replace('|','_')}_profiles.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[saved] {out}")
    plt.close(fig)


def plot_contingency_heatmap(cont_row_norm, title):

    outdir = os.path.join(os.getcwd(), "kmeans_figs")


    order = cont_row_norm.sum(1).sort_values(ascending=False).index
    cont_row_norm = cont_row_norm.loc[order]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(cont_row_norm.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    for i in range(cont_row_norm.shape[0]):
        for j in range(cont_row_norm.shape[1]):
            ax.text(j, i, f"{cont_row_norm.iloc[i,j]:.2f}", ha='center', va='center', fontsize=9)
    ax.set_yticks(range(cont_row_norm.shape[0])); ax.set_yticklabels(cont_row_norm.index)
    ax.set_xticks(range(cont_row_norm.shape[1])); ax.set_xticklabels(cont_row_norm.columns)
    ax.set_xlabel('Cluster'); ax.set_ylabel('Label'); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized proportion')
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{title.replace(' ', '_').replace('|','_')}_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[saved] {out}")
    plt.close(fig)


def plot_silhouette_bars(X_for_cluster, labels, title):

    outdir = os.path.join(os.getcwd(), "kmeans_figs")

    s_all = silhouette_samples(X_for_cluster, labels)
    df = pd.DataFrame({'s': s_all, 'cluster': labels})
    order = sorted(df['cluster'].unique())
    means = df.groupby('cluster')['s'].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(5.6,3.6))
    ax.bar(range(len(order)), means.values)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f'C{k}' for k in order])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean silhouette'); 
    ax.set_title(title + f" | overall={silhouette_score(X_for_cluster, labels):.2f}")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{title.replace(' ', '_').replace('|','_')}_silhouette.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[saved] {out}")
    plt.close(fig)

def result_panel(
    data_dict, angle_type, conditions,  
    k,                                   
    group_map=None,                      
    axes=('Y'), frame_len=201, fs=200,
    center_mode='subject_mean',
    use_row_l2=False, pca_var=0.95, random_state=0
):

    if isinstance(axes, str):
        axes = (axes,)
    else:
        axes = tuple(axes)


    X_raw, meta = build_subject_condition_matrix(
        data_dict, angle_type=angle_type, axes=axes, frame_len=frame_len,
        pad_mode='edge', conditions=conditions
    )

    if group_map is not None:
        meta = meta.copy()
        meta['Label'] = meta['Condition'].map(group_map)
        keep = meta['Label'].notna().values
        X_raw, meta = X_raw.iloc[keep].reset_index(drop=True), meta.iloc[keep].reset_index(drop=True)
    else:
        meta = meta.copy()
        meta['Label'] = meta['Condition']

    
    X_centered = center_waveforms_by_subject(X_raw, meta, axes=axes, frame_len=frame_len, mode=center_mode)
    X_std, X_for_cluster, scaler, row_norm, pca = standardize_pca_for_waveforms(
        X_centered, use_row_l2=use_row_l2, pca_var=pca_var, random_state=random_state
    )

    km = KMeans(n_clusters=int(k), n_init=10, random_state=random_state)
    labels = km.fit_predict(X_for_cluster)


    y_codes, y_uni = pd.factorize(meta['Label'])
    cont = pd.crosstab(meta['Label'], pd.Series(labels, name='Cluster'))
    row_norm_tbl = (cont.T / cont.sum(1)).T.fillna(0.0)
    purity = (cont / cont.sum(0)).max(0).fillna(0.0)
    ari = adjusted_rand_score(y_codes, labels)
    nmi = normalized_mutual_info_score(y_codes, labels)
    ami = adjusted_mutual_info_score(y_codes,labels, average_method='arithmetic')
    h, c, v = homogeneity_completeness_v_measure(y_codes, labels)
    try:
        chi2, pval, dof, _ = chi2_contingency(cont.values)
    except Exception:
        chi2, pval, dof = np.nan, np.nan, np.nan
    rind, cind = linear_sum_assignment(cont.values.max() - cont.values)
    best_acc = cont.values[rind, cind].sum() / cont.values.sum() if cont.values.sum()>0 else np.nan
    mapping = dict(zip(cind, cont.index[rind]))

    print(f"\n ARI={ari:.3f},NMI={nmi:.3f},AMI={ami:.3f}, H={h:.3f}, C={c:.3f}, V={v:.3f}")
    print(f" χ²={chi2:.2f}, dof={dof}, p={pval:.3g}")
    print(f" best一一match accuracy={best_acc:.3f}, mapping cluster→label: {mapping}")
    print(f"purity :\n{purity.round(3)}")
    print(f"\nContingency (Label × Cluster):\n{cont}")


    try:
        _p_viz = PCA(n_components=2, random_state=random_state)
        X_viz = _p_viz.fit_transform(X_std)  
        evr_pair = getattr(_p_viz, 'explained_variance_ratio_', np.array([np.nan, np.nan]))
        pca_scatter_with_ellipses(
            X_viz,
            labels_for_color=np.array(labels),
            label_names_for_ellipse=np.array(meta['Label']),
            title=f"K-means Clustering of {angle_type} angles",
            evr_pair=evr_pair
        )
    except Exception as e:
        print(f"[warn] PCA 可视化跳过：{e}")

    plot_contingency_heatmap(row_norm_tbl, title=f"Label × Cluster (row-normalized) | ARI={ari:.2f}")

    plot_cluster_profiles(
        X_centered, np.array(labels), axes=axes, frame_len=frame_len, fs=fs,
        title=f"Cluster profiles | {angle_type} | k={k}"
    )


    plot_silhouette_bars(X_for_cluster, np.array(labels), title=f"Silhouette | {angle_type} | k={k}")

    return dict(
        labels=labels, meta=meta, X_centered=X_centered, X_for_cluster=X_for_cluster,
        contingency=cont, row_norm=row_norm_tbl,
        metrics=dict(ARI=ari, NMI=nmi, H=h, C=c, V=v, chi2=chi2, p=pval, mapping=mapping,
                     purity=purity, best_acc=best_acc)
    )

ROOT = '/Users/wangtingyu/Desktop/LMB/data/V3D_new_processing'
data_dict = build_data_dict(ROOT)


ANGLE_TYPES = ['HFSHK', 'FFHF']

CONDITIONS_3 = ['WW-L-0_S', 'WW-N-0_S', 'WOW-L-nR_S', 'WOW-N-nR_S', 'WOW-L-R_S']
GROUP_MAP = {
    'WW-L-0_S':   'WW',
    'WW-N-0_S':   'WW',
    'WOW-L-nR_S': 'WOW-nR',
    'WOW-N-nR_S': 'WOW-nR',
    'WOW-L-R_S':  'WOW-R',
}


K_THREE = 3         
CENTER_MODE = 'subject_mean'  # 'subject_mean' or 'none'
USE_ROW_L2 = False   
PCA_VAR = 0.95       
SEED = 42
colors = ['#54B345','#FA7F6F','#2878B5']

def _main_standing_ancova_original():
    data_dict = build_data_dict(ROOT)

    angle_type = 'FFHF'
    axis = 'Y'

    conditions = [
        ('WOW-L-nR_S', 'WOW-N-nR_S'),
        ('WOW-L-R_S',  'WOW-N-nR_S'),
        ('WW-L-0_S',   'WW-N-0_S'),
    ]

    COV_COND = ('B-L-0_S', 'B-N-0_S')

    df_long = prepare_ancova(
        data_dict,
        conditions=conditions,
        cov_condition=COV_COND,
        angle_type=angle_type,
        axis=axis
    )
    _ = run_ancova_with_homogeneity(df_long, alpha=0.05)


def _main_standing_panel3_original():
    data_dict = build_data_dict(ROOT)

    for angle in ANGLE_TYPES:
        print(f"\n================ Angle: {angle} | 三大类 (k={K_THREE}) ================")
        panel3 = result_panel(
            data_dict=data_dict,
            angle_type=angle,
            conditions=CONDITIONS_3,
            k=K_THREE,
            group_map=GROUP_MAP,
            axes=('Y',),
            center_mode=CENTER_MODE,
            use_row_l2=USE_ROW_L2,
            pca_var=PCA_VAR,
            random_state=SEED
        )

def main():
    parser = argparse.ArgumentParser(
        description="Standing analysis dispatcher (ancova | panel3)"
    )
    parser.add_argument(
        "--analysis",
        choices=["ancova", "panel3"],
        required=True,
        help="Choose which original main to run"
    )
    args = parser.parse_args()

    if args.analysis == "ancova":
        _main_standing_ancova_original()
    elif args.analysis == "panel3":
        _main_standing_panel3_original()


if __name__ == "__main__":
    main()