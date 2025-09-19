
import os, re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spm1d

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


ROOT_PATH       = "/Users/wangtingyu/Desktop/LMB/data/V3D_new_processing"
TARGET_FILENAME = "Gait_Right_All_Patterns_Full_Cycle.txt"
SUBJECTS        = range(1, 16+1)   
SKIPROWS        = 5
CLEAN_COND_TAIL = True            
RESAMPLE_TO     = 101            
FIGSIZE_MSD = (8, 4)  

CONDITIONS = ["WOW-L-R_G", "WOW-L-nR_G", "WW-L-0_G"]

# 分析哪些角度/轴
ANGLES = ["FFHF", "HFSHK"]
AXES   = ["Y"]

# 是否把“每个 trial 当一条观测”喂给 anova1rm（推荐=True）
USE_TRIAL_LEVEL_RM = True

ALPHA = 0.05
# ====================================================


# ---------------- 工具函数（解析/清洗） ----------------
def _last_seg(s: str) -> str:
    return re.split(r"[\\/]", s.strip())[-1]

def _extract_condition(cell: str) -> str:
    m = re.search(r"\b[A-Z]{1,4}-[A-Z]-\d+_[SG]\b", cell)
    cond = m.group(0) if m else (_last_seg(cell)[4:-7] if len(_last_seg(cell)) >= 11 else _last_seg(cell))
    if CLEAN_COND_TAIL:
        cond = re.sub(r'(_\d+\.c$)', '', cond)
    return cond

def _extract_angle(cell: str) -> str:
    seg = _last_seg(cell)
    m = re.search(r"(?:Angles|Angle|Patterns|Pattern)_(.+?)(?:[_\.]|$)", seg, flags=re.I)
    if m: return m.group(1)
    m = re.search(r"\b(FFHF|FFMF|MFHF|HFSHK|HLXFF|HFAF|Knee|Ankle|Hip|Shank|Thigh|Foot|Pelvis|Spine|Rearfoot|Forefoot)\b",
                  seg, flags=re.I)
    if m: return m.group(1).upper()
    return seg[6:-6] if len(seg) >= 12 else seg

def _align_header_to_data(header_cells: List[str], n_data_cols: int) -> List[str]:
    if len(header_cells) == n_data_cols:
        return header_cells
    if len(header_cells) == n_data_cols + 1:
        return header_cells[1:]
    if len(header_cells) > n_data_cols:
        return header_cells[:n_data_cols]
    return header_cells + [""] * (n_data_cols - len(header_cells))

def _resample(vec: List[float], target: int) -> List[float]:
    vec = list(map(float, vec))
    if (target is None) or (len(vec) == target):
        return vec
    x  = np.linspace(0, 1, len(vec))
    xi = np.linspace(0, 1, target)
    return np.interp(xi, x, vec).tolist()

def _mean_waveform(trials: List[List[float]]) -> np.ndarray:
    arr = np.array(trials, float)
    return np.nanmean(arr, axis=0)


# ---------------- 读取：构建 data_dict ----------------
def build_full_data_dict(root_path: str, target_filename: str, subjects) -> Dict:
    """
    data_dict[subject][condition][angle]['X'|'Y'|'Z'] -> List[List[float]]
    """
    data_dict: Dict[str, Dict] = {}
    for i in subjects:
        folder = f"S{i:02d}"
        file_path = os.path.join(root_path, folder, "Output", target_filename)
        if not os.path.exists(file_path):
            print(f"[skip] {folder} 缺文件：{file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            line0 = f.readline().strip().split("\t")  # conditions
            line1 = f.readline().strip().split("\t")  # angles

        raw  = pd.read_csv(file_path, sep="\t", header=None, skiprows=SKIPROWS, encoding="utf-8")
        data = raw.iloc[:, 1:].copy().ffill().bfill()

        n_cols = data.shape[1]
        if n_cols % 3 != 0:
            print(f"[warn] {folder}: 数据列 {n_cols} 不是 3 的倍数，可能缺列。")

        cond_cells  = _align_header_to_data(line0, n_cols)
        angle_cells = _align_header_to_data(line1, n_cols)
        conditions  = list(map(_extract_condition, cond_cells))
        angles      = list(map(_extract_angle,     angle_cells))

        for col_idx in range(0, (n_cols // 3) * 3, 3):
            condition  = conditions[col_idx]
            angle_type = angles[col_idx]

            # 历史特殊修正（若需要）
            if folder == "S07" and condition.startswith("WW-A-0_S"):
                condition = "WW-A-0_S"

            xs = _resample(data.iloc[:, col_idx    ].tolist(), RESAMPLE_TO)
            ys = _resample(data.iloc[:, col_idx + 1].tolist(), RESAMPLE_TO)
            zs = _resample(data.iloc[:, col_idx + 2].tolist(), RESAMPLE_TO)

            node = data_dict.setdefault(folder, {}).setdefault(condition, {}).setdefault(angle_type, {})
            node.setdefault('X', []).append(xs)
            node.setdefault('Y', []).append(ys)
            node.setdefault('Z', []).append(zs)

    return data_dict


def plot_gait_cond_mean_sd(
    data_dict,
    angle_type,
    conditions,
    axis='Y',
    # 文本内容
    title=None,
    xlabel='Gait cycle (%)',
    ylabel=None,
    # 文本字号
    label_fs=12,
    tick_fs=10,
    title_fs=15,
    legend_fs=10,
    # 图形外观
    figsize=(8, 4),
    fill_alpha=0.18,
    legend_ncol=1,
    legend_loc='best',
    # 颜色 & 图例
    color_map=None,
    legend_labels=None,

    include_n=True,
    # 线型/填充
    line_kw=None,
    fill_kw=None,
    # ====== 新增：保存参数 ======
    save_path: str | None = None,   # 例如 "/path/to/fig.tiff"
    save_dpi: int = 300
):

    line_kw = {} if line_kw is None else dict(line_kw)
    fill_kw = {} if fill_kw is None else dict(fill_kw)

    # 自动识别目标长度 L
    L = None
    for subj in data_dict:
        for cond in conditions:
            trials = data_dict.get(subj, {}).get(cond, {}).get(angle_type, {}).get(axis)
            if trials:
                L = len(trials[0]); break
        if L is not None: break
    if L is None:
        raise ValueError("没有找到任何匹配的波形：检查 angle_type / conditions / axis。")

    def _resample(y, L):
        y = np.asarray(y, float)
        if y.size == L: return y
        x  = np.linspace(0, 1, y.size)
        xi = np.linspace(0, 1, L)
        return np.interp(xi, x, y)

    x_pct = np.linspace(0, 100, L)
    stats = {}

    fig, ax = plt.subplots(figsize=figsize)
    for i, cond in enumerate(conditions):
        # 颜色
        color = None
        if isinstance(color_map, dict):
            color = color_map.get(cond, None)
        elif isinstance(color_map, (list, tuple)) and len(color_map) > 0:
            color = color_map[i % len(color_map)]

        # 被试内 trial 均值 → 跨被试统计
        subj_means = []
        for subj in sorted(data_dict.keys()):
            trials = data_dict.get(subj, {}).get(cond, {}).get(angle_type, {}).get(axis)
            if not trials: continue
            trials_rs = [_resample(t, L) for t in trials]
            subj_means.append(np.mean(np.vstack(trials_rs), axis=0))
        if not subj_means:
            print(f"[warn] 条件 {cond} 无数据，跳过。"); continue

        A = np.vstack(subj_means)
        m = A.mean(axis=0)
        s = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(L)
        stats[cond] = {'mean': m, 'std': s, 'n_subj': A.shape[0], 'L': L}

        # 图例文字
        base_label = legend_labels.get(cond, cond) if isinstance(legend_labels, dict) else cond
        label = f"{base_label} (n={A.shape[0]})" if include_n else base_label

        ax.plot(x_pct, m, label=label, color=color, **line_kw)
        ax.fill_between(x_pct, m - s, m + s, alpha=fill_alpha, color=color, **fill_kw)

    # 标签/标题/图例
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel if ylabel is not None else f"{angle_type}-{axis}", fontsize=label_fs)
    ax.set_title(title or f"Mean ± SD | {angle_type}-{axis}", fontsize=title_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.legend(ncol=legend_ncol, loc=legend_loc, prop={'size': legend_fs})

    # 横坐标拉满
    ax.set_xlim(x_pct[0], x_pct[-1])
    ax.margins(x=0)
    ax.autoscale(enable=True, axis='x', tight=True)

    plt.tight_layout()

    # ====== 保存为 TIFF (dpi=save_dpi) ======
    if save_path:
        # 自动补后缀 & 创建目录
        root, ext = os.path.splitext(save_path)
        if ext.lower() not in ('.tif', '.tiff'):
            save_path = root + '.tiff'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=save_dpi, format='tiff', bbox_inches='tight')
        print(f"[saved] {save_path} (dpi={save_dpi})")

    plt.show()
    return stats


def build_rm_longform(
    data_dict: Dict, angle: str, axis: str, cond_list: List[str], use_trial_level: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:

    G = len(cond_list)
    subj_order, cond_means_lists = [], {c: [] for c in cond_list}
    YL, groups, subs = [], [], []
    target_len = None

    for subj in sorted(data_dict.keys()):
        all_trials = []
        ok = True
        for cond in cond_list:
            trials = data_dict[subj].get(cond, {}).get(angle, {}).get(axis)
            if not trials:
                ok = False; break
            if target_len is None:
                target_len = len(trials[0])
            if any(len(t) != target_len for t in trials):
                ok = False; break
            all_trials.append((cond, trials))
        if not ok or len(all_trials) != G:
            continue

        subj_order.append(subj)

        # 每条件 trial-mean（用于配对事后 t）
        for cond, trials in all_trials:
            cond_means_lists[cond].append(_mean_waveform(trials))

        # 长表
        if use_trial_level:
            for g_idx, (cond, trials) in enumerate(all_trials):
                for tr in trials:
                    YL.append(tr); groups.append(g_idx); subs.append(subj)
        else:
            for g_idx, (cond, trials) in enumerate(all_trials):
                YL.append(_mean_waveform(trials)); groups.append(g_idx); subs.append(subj)

    if not subj_order:
        raise RuntimeError("没有满足‘每被试各条件都有数据且长度一致’的被试。")

    YL = np.vstack(YL)
    group = np.array(groups, int)
    subjects = np.array(subs, object)
    cond_means = {c: np.vstack(cond_means_lists[c]) for c in cond_list}
    return YL, group, subjects, subj_order, cond_means

def rm_anova_spm1d_compat(
    YL, group, subjects, cond_list,
    alpha=0.05,
    title=None,
    xlabel='Stance phase (%)',
    ylabel='SPM{F}',
    title_fs=15,
    label_fs=12,
    tick_fs=10,
    figsize=FIGSIZE_MSD,
):
    import numpy as np, matplotlib.pyplot as plt, spm1d

    # ---- helpers ----
    def _format_p(p):
        if p is None:
            return r"$\mathit{P}$=n/a"
        return r"$\mathit{P}$<0.001" if p < 0.001 else rf"$\mathit{{P}}$={p:.3f}"

    def _int_endpoints(cl, n_time):
        i0, i1 = cl.endpoints
        i0 = int(np.floor(i0)); i1 = int(np.ceil(i1))
        i0 = max(0, min(i0, n_time-1)); i1 = max(i0+1, min(i1, n_time))
        return i0, i1

    def _annotate_peak(ax, x, y, text, prefer_above=True):
        ymin, ymax = ax.get_ylim(); yspan = ymax - ymin
        # 增加上下余量以避免重叠
        ax.set_ylim(ymin - 0.06*yspan, ymax + 0.18*yspan)
        ymin2, ymax2 = ax.get_ylim(); yspan2 = ymax2 - ymin2
        ytxt = y + (0.06*yspan2 if prefer_above else -0.06*yspan2)
        # 夹紧到安全区
        ytxt = min(max(ytxt, ymin2 + 0.04*yspan2), ymax2 - 0.04*yspan2)
        va = "bottom" if ytxt >= y else "top"
        ax.annotate(text, xy=(x, y), xytext=(x, ytxt),
                    ha="center", va=va, fontsize=10,
                    arrowprops=dict(arrowstyle="->", lw=1),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="0.5"))

    # 映射被试 → 整数
    subj_levels = np.unique(subjects)
    subj_map = {s:i for i,s in enumerate(subj_levels)}
    subj_idx = np.array([subj_map[s] for s in subjects], int)

    # 统计与推断
    model = spm1d.stats.anova1rm(YL, group.astype(int), subj_idx)
    spmi  = model.inference(alpha=alpha)

    # 出图
    fig, ax = plt.subplots(figsize=figsize)
    spmi.plot(ax=ax)
    spmi.plot_threshold_label(ax=ax)

    # 轴格式
    n_time = YL.shape[1]
    tick_pos = np.linspace(0, n_time-1, 6)
    ax.set_xticks(tick_pos); ax.set_xticklabels([f"{p:.0f}" for p in np.linspace(0,100,6)])
    ax.set_xlim(0, n_time-1); ax.margins(x=0)
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_title(title or f"SPM{{F}} RM-ANOVA (α={alpha})", fontsize=title_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)

    # 对每个显著簇做标注（F 统计量为单侧正向）
    z   = np.asarray(spmi.z).ravel()
    thr = float(getattr(spmi, 'zstar', np.nan))
    clusters = getattr(spmi, "clusters", [])
    if clusters:
        for cl in clusters:
            i0, i1 = _int_endpoints(cl, z.size)
            seg = z[i0:i1]
            # 取阈值以上的峰值位置；若阈值不可用/没找到，则退化为区间内最大值
            over = np.where(seg >= thr - 1e-12)[0] if np.isfinite(thr) else np.array([], int)
            j_rel = over[np.argmax(seg[over])] if over.size else int(np.argmax(seg))
            j     = i0 + j_rel
            zpk   = float(seg[j_rel])
            _annotate_peak(ax, j, zpk, f"F={zpk:.2f}\n{_format_p(cl.P)}", prefer_above=True)
    else:
        # 无显著：标注全局最大 F 值并注明 n.s.
        j = int(np.argmax(z)); zpk = float(z[j])
        _annotate_peak(ax, j, zpk, f"F={zpk:.2f}\n(n.s.)", prefer_above=True)

    fig.tight_layout()
    return spmi, fig, ax

def posthoc_pairwise_rm_means(
    cond_means, cond_list,
    alpha=0.05, two_tailed=True,
    plot=False, save_dir=None, angle=None, axis=None,
    title_map=None,                   # 例：{('WOW-L-nR_G','WW-L-0_G'): '你的标题', ...}
    figsize=(8, 4.6),                 # 和 Mean±SD 一致
    xlabel='Stance phase (%)', ylabel='SPM{t}',
    title_fs=16, label_fs=14, tick_fs=12,
    save_dpi=300,
):
    import os, numpy as np, matplotlib.pyplot as plt, spm1d

    # ---------- helpers ----------
    def _format_p(p):
        # 大写+斜体；<0.001 用门槛显示
        return r"$\mathit{P}$<0.001" if (p is not None and p < 0.001) else rf"$\mathit{{P}}$={p:.3f}"

    def _int_endpoints(cl, n_time):
        i0, i1 = cl.endpoints
        i0 = int(np.floor(i0)); i1 = int(np.ceil(i1))
        i0 = max(0, min(i0, n_time-1)); i1 = max(i0+1, min(i1, n_time))
        return i0, i1

    def _annotate_peak(ax, x, y, text, prefer_above=True):
        # 增加上下余量并夹紧，避免文本跑出画布
        ymin, ymax = ax.get_ylim(); yspan = ymax - ymin
        ax.set_ylim(ymin - 0.06*yspan, ymax + 0.18*yspan)
        ymin2, ymax2 = ax.get_ylim(); yspan2 = ymax2 - ymin2
        ytxt = y + (0.06*yspan2 if prefer_above else -0.06*yspan2)
        ytxt = min(max(ytxt, ymin2 + 0.04*yspan2), ymax2 - 0.04*yspan2)
        va = "bottom" if ytxt >= y else "top"
        ax.annotate(
            text, xy=(x, y), xytext=(x, ytxt),
            ha="center", va=va, fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="0.5"),
        )

    if plot and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    for i in range(len(cond_list)):
        for j in range(i+1, len(cond_list)):
            ci, cj = cond_list[i], cond_list[j]
            Yi, Yj = cond_means[ci], cond_means[cj]   # (n_subj, n_time)

            # 统计 + 推断
            t = spm1d.stats.ttest_paired(Yi, Yj)
            spmi = t.inference(alpha=alpha, two_tailed=two_tailed)

            # 记录最小簇p（保持你的返回结构不变）
            minp = float(min([cl.P for cl in getattr(spmi, "clusters", [])], default=1.0))
            results.append((f"{cj} - {ci}", spmi, minp))

            # —— 只在 plot=True 时出图并标注 —— #
            if plot:
                fig, ax = plt.subplots(figsize=figsize)
                spmi.plot(ax=ax)
                spmi.plot_threshold_label(ax=ax)

                # 轴：0–100%
                n_time = Yi.shape[1]
                ax.set_xlim(0, n_time-1); ax.margins(x=0)
                xt = np.linspace(0, n_time-1, 6)
                ax.set_xticks(xt); ax.set_xticklabels([f"{p:.0f}" for p in np.linspace(0, 100, 6)])

                # 标题（优先用映射）
                ttl = None
                if title_map:
                    ttl = title_map.get((ci, cj)) or title_map.get((cj, ci))
                if ttl is None:
                    ttl = f"SPM{{t}} posthoc | {angle}-{axis} | {cj} vs {ci}"
                ax.set_title(ttl, fontsize=title_fs)

                ax.set_xlabel(xlabel, fontsize=label_fs)
                ax.set_ylabel(ylabel, fontsize=label_fs)
                ax.tick_params(axis='both', labelsize=tick_fs)

                # —— 对每个显著簇做一次“阈值内真正峰值”的标注 —— #
                z   = np.asarray(spmi.z).ravel()
                thr = float(getattr(spmi, 'zstar', np.nan))
                clusters = getattr(spmi, "clusters", [])
                if clusters:
                    for cl in clusters:
                        i0, i1 = _int_endpoints(cl, z.size)
                        seg = z[i0:i1]

                        # 判断簇整体符号（也可用 seg.sum() 的符号）
                        is_pos = (np.mean(seg) >= 0)

                        # 在超过阈值的索引里找峰值（防止边界处出现无关小峰）
                        eps = 1e-12
                        if is_pos:
                            idx = np.where(seg >=  thr - eps)[0]
                            j_rel = idx[np.argmax(seg[idx])] if idx.size else int(np.argmax(seg))
                        else:
                            idx = np.where(seg <= -thr + eps)[0]
                            j_rel = idx[np.argmin(seg[idx])] if idx.size else int(np.argmin(seg))

                        j_glb = i0 + j_rel
                        zpk   = float(seg[j_rel])
                        _annotate_peak(ax, j_glb, zpk, f"t={zpk:.2f}\n{_format_p(cl.P)}", prefer_above=(zpk>=0))
                else:
                    # 无显著簇：标注全局 |t| 最大处，并显示 n.s.
                    jg = int(np.argmax(np.abs(z))); zpk = float(z[jg])
                    _annotate_peak(ax, jg, zpk, f"t={zpk:.2f}\n(n.s.)", prefer_above=(zpk>=0))

                plt.tight_layout()
                if save_dir:
                    fname = f"posthoc_{angle}-{axis}_{cj}_vs_{ci}.tiff".replace("/", "-")
                    path = os.path.join(save_dir, fname)
                    plt.savefig(path, dpi=save_dpi, format='tiff', bbox_inches='tight')
                    print(f"[saved] {path}  dpi={save_dpi}")
                plt.show()
                plt.close(fig)

    return results


# ---------------- 打印显著区段（F / t）含端点安全修复 ----------------
def _int_endpoints(cl, n_time: int):
    """把 cluster.endpoints 转成整数区间 [i0, i1) 并裁剪到 [0,n_time]。"""
    i0, i1 = cl.endpoints
    i0 = int(np.floor(i0))
    i1 = int(np.ceil(i1))
    i0 = max(0, min(i0, n_time-1))
    i1 = max(i0+1, min(i1, n_time))
    return i0, i1

def report_f_clusters(spmi, label=""):
    z = np.asarray(spmi.z).ravel()
    n_time = z.size
    clusters = getattr(spmi, "clusters", [])
    if not clusters:
        print(f"{label}: 无显著区段"); return
    print(f"{label}: 显著区段（SPM{{F}}，单尾）")
    for k, cl in enumerate(clusters, 1):
        i0, i1 = _int_endpoints(cl, n_time)
        zpeak = float(z[i0:i1].max())
        s_pct = i0/(n_time-1)*100.0
        e_pct = (i1-1)/(n_time-1)*100.0
        print(f"  簇{k}: {s_pct:.1f}% → {e_pct:.1f}%,  p={cl.P:.4f},  F峰值={zpeak:.2f}")

def report_t_clusters(spmi, label=""):
    z = np.asarray(spmi.z).ravel()
    n_time = z.size
    clusters = getattr(spmi, "clusters", [])
    if not clusters:
        print(f"{label}: 无显著区段"); return
    print(f"{label}: 显著区段（SPM{{t}}，两尾）")
    for k, cl in enumerate(clusters, 1):
        i0, i1 = _int_endpoints(cl, n_time)
        seg = z[i0:i1]
        pos, neg = float(seg.max()), float(seg.min())
        zpeak = pos if abs(pos) >= abs(neg) else neg
        direction = "正簇(后者>前者)" if zpeak > 0 else "负簇(前者>后者)"
        s_pct = i0/(n_time-1)*100.0
        e_pct = (i1-1)/(n_time-1)*100.0
        print(f"  簇{k}: {direction}, {s_pct:.1f}% → {e_pct:.1f}%,  p={cl.P:.4f},  t峰值={zpeak:.2f}")


def ancova_spm1d_two_groups(
    data_dict,
    angle="HFSHK",
    axis="Y",
    condA="WW-N-0_G",           # 组A（基准，编码为0）
    condB="WOW-N-nR_G",         # 组B（编码为1）
    cov_cond="B-0-0_G",         # 协变量条件（每被试取均值做标量协变量）
    alpha=0.05,
    save_dir="/Users/wangtingyu/Desktop/LMB/output/spm1d_ancova",
    title=None,
    figsize=(8,4),
    title_fs=13, label_fs=12, tick_fs=10,
    dpi=300,
    direction="B-A"             # "B-A": t>0 表示 B>A；"A-B": t>0 表示 A>B
):
    import os, numpy as np, matplotlib.pyplot as plt, spm1d
    os.makedirs(save_dir, exist_ok=True)

    # —— 兼容 spm1d 不同 GLM API（类式/函数式） ——
    def _glm_contrast_inference(Y, X, c, alpha=0.05, two_tailed=True):
        glm_api = spm1d.stats.glm
        if hasattr(glm_api, "OLS"):                 # 类式
            mdl = glm_api.OLS(Y, X)
            return mdl.contrast(c).inference(alpha=alpha, two_tailed=two_tailed)
        else:                                       # 函数式
            return glm_api(Y, X, c).inference(alpha=alpha, two_tailed=two_tailed)

    # —— 安全端点：将簇端点夹紧到 [0, n)；若无 endpoints，尝试 indices/I 退化处理 ——
    def _int_endpoints(cl, n):
        e = getattr(cl, "endpoints", None)
        if e is not None:
            i0, i1 = int(e[0]), int(e[1])
            i0 = max(0, min(i0, n-1))
            i1 = max(i0+1, min(int(i1), n))
            return i0, i1
        # 回退：从索引集合恢复区间（右开）
        idx = getattr(cl, "indices", getattr(cl, "I", None))
        if idx is None:
            return 0, 0
        idx = np.asarray(idx).ravel()
        if idx.size == 0:
            return 0, 0
        i0 = int(idx.min())
        i1 = int(idx.max()) + 1
        i0 = max(0, min(i0, n-1))
        i1 = max(i0+1, min(i1, n))
        return i0, i1

    # —— P 值格式：大写+斜体；P<0.001 时用阈值写法 ——
    def _fmt_P(p):
        return r"$P<0.001$" if (p is not None and p < 0.001) else rf"$P={p:.3f}$"

    # —— 峰值标注（自动留白，避免顶到标题） ——
    def _annotate_peak(ax, x, y, text, prefer_above=True):
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        ax.set_ylim(ymin - 0.06*yspan, ymax + 0.18*yspan)  # 加上下留白
        ymin2, ymax2 = ax.get_ylim()
        yspan2 = ymax2 - ymin2
        offset = 0.06 * yspan2
        ytxt = y + (offset if prefer_above else -offset)
        ytxt = min(max(ytxt, ymin2 + 0.04*yspan2), ymax2 - 0.04*yspan2)
        va = "bottom" if ytxt >= y else "top"
        ax.annotate(
            text, xy=(x, y), xytext=(x, ytxt),
            ha="center", va=va, fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="0.5")
        )

    # —— 聚合数据（仅保留三者都齐的被试） ——
    subjects_ok, Y_rows, G_vec, C_vec = [], [], [], []
    L = None
    for subj in sorted(data_dict.keys()):
        nodeA = data_dict.get(subj, {}).get(condA, {}).get(angle, {}).get(axis)
        nodeB = data_dict.get(subj, {}).get(condB, {}).get(angle, {}).get(axis)
        nodeC = data_dict.get(subj, {}).get(cov_cond, {}).get(angle, {}).get(axis)
        if not (nodeA and nodeB and nodeC):
            continue
        if L is None: L = len(nodeA[0])
        if len(nodeB[0]) != L or len(nodeC[0]) != L:
            continue

        YA = np.mean(np.vstack(nodeA), axis=0)                # trial-mean
        YB = np.mean(np.vstack(nodeB), axis=0)
        C  = float(np.mean(np.mean(np.vstack(nodeC), axis=0)))  # 协变量标量

        subjects_ok.append(subj)
        Y_rows.append(YA); G_vec.append(0); C_vec.append(C)
        Y_rows.append(YB); G_vec.append(1); C_vec.append(C)

    if not subjects_ok:
        raise RuntimeError("没有找到同时具备 condA/condB/cov_cond 的被试。")

    Y = np.vstack(Y_rows)              # (n_obs, L)
    G = np.array(G_vec, float)         # 0/1
    C = np.array(C_vec, float); C -= C.mean()

    # —— 交互检验：Y ~ 1 + G + C + G*C，检验 β(G*C)=0 ——
    X_int  = np.column_stack([np.ones(Y.shape[0]), G, C, G*C])
    c_int  = np.array([0, 0, 0, 1.0])
    ti_int = _glm_contrast_inference(Y, X_int, c_int, alpha=alpha, two_tailed=True)
    interaction_sig = (len(getattr(ti_int, "clusters", [])) > 0)

    # —— ANCOVA 主效应：Y ~ 1 + G + C，检验 β(G)=0（方向可控） ——
    X  = np.column_stack([np.ones(Y.shape[0]), G, C])
    sign = +1.0 if direction.upper()=="B-A" else -1.0
    c   = np.array([0, sign, 0])       # t>0 表示 (B-A)>0 或 (A-B)>0
    ti  = _glm_contrast_inference(Y, X, c, alpha=alpha, two_tailed=True)  # ← 这是推断对象（spmi）

    # —— 作图 ——
    fig, ax = plt.subplots(figsize=figsize)
    ti.plot(ax=ax)
    ti.plot_threshold_label(ax=ax)

    n_time = Y.shape[1]
    ax.set_xlim(0, n_time-1); ax.margins(x=0)
    xt = np.linspace(0, n_time-1, 6)
    ax.set_xticks(xt); ax.set_xticklabels([f"{p:.0f}" for p in np.linspace(0, 100, 6)])

    if title is None:
        comp = f"{condB} vs {condA}" if direction.upper()=="B-A" else f"{condA} vs {condB}"
        title = f"ANCOVA of {angle}: {comp} (covariate: {cov_cond} mean)"
        if interaction_sig: title += "  [interaction detected]"
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("Stance phase (%)", fontsize=label_fs)
    ax.set_ylabel("SPM{t}", fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)

    # —— 峰值标注（*P* 大写/斜体；P<0.001） ——
    zt = np.asarray(ti.z).ravel()
    clusters = getattr(ti, "clusters", [])
    if clusters:
        for cl in clusters:
            i0, i1 = _int_endpoints(cl, zt.size)
            seg = zt[i0:i1]
            j_rel = int(np.argmax(np.abs(seg)))
            j = i0 + j_rel
            zpk = float(seg[j_rel])
            Ptxt = _fmt_P(cl.P)
            _annotate_peak(ax, j, zpk, f"t={zpk:.2f}\n{Ptxt}", prefer_above=(zpk>=0))
    else:
        j = int(np.argmax(np.abs(zt))); zpk = float(zt[j])
        _annotate_peak(ax, j, zpk, "n.s.", prefer_above=(zpk>=0))

    fig.tight_layout()
    fname = f"ANCOVA_{angle}-{axis}_{condB}_vs_{condA}_cov_{cov_cond}.tiff".replace("/", "-")
    outp = os.path.join(save_dir, fname)
    fig.savefig(outp, dpi=dpi, format="tiff", bbox_inches="tight")
    plt.close(fig)

    # —— 控制台信息 —— 
    print(f"[ANCOVA] subjects={len(subjects_ok)}  obs={Y.shape[0]}  L={L}")
    if interaction_sig:
        print("  ⚠️ 发现显著的组×协变量交互（同质斜率假设被破坏）→ 主效应需谨慎解释。")
    else:
        print("  ✅ 组×协变量交互不显著 → 满足同质斜率假设。")
    print(f"  saved: {outp}")

    # ★ 新增：打印显著簇的明细（与 RM-ANOVA 一致的风格） ★
    z = np.asarray(getattr(ti, "z", []), dtype=float).ravel()
    cl_list = getattr(ti, "clusters", [])
    if cl_list:
        print(f"[clusters] {angle}-{axis} (ANCOVA): 共 {len(cl_list)} 个显著簇")
        for k, cl in enumerate(cl_list, 1):
            i0, i1 = _int_endpoints(cl, len(z))
            extent = getattr(cl, "extent", i1 - i0)
            seg = z[i0:i1] if (i1 > i0 and len(z) > 0) else np.array([], dtype=float)
            zpeak = float(seg[np.argmax(np.abs(seg))]) if seg.size else float("nan")
            Pval = getattr(cl, "P", None)
            Ptxt = f"{Pval:.4f}" if (Pval is not None) else "n/a"
            print(f"  簇{k}: 区间=({i0}, {i1})  长度={extent}  峰值t={zpeak:.2f}  P={Ptxt}")
    else:
        print(f"[clusters] {angle}-{axis} (ANCOVA): 无显著簇 (n.s.)")

    # ★ 返回时带出推断对象（命名为 spmi 以便主程序可直接使用） ★
    return dict(
        spmi=ti,                     # 推断后的 SPM 对象（含 .clusters / .zstar / .z）
        spmi_interaction=ti_int,     # 交互项推断对象
        interaction_sig=interaction_sig,
        subjects=subjects_ok,
        out_path=outp
    )


if __name__ == "__main__":
   
    print("spm1d version:", getattr(spm1d, "__version__", "unknown"))


    data_dict = build_full_data_dict(ROOT_PATH, TARGET_FILENAME, SUBJECTS)
   

    # 2) ANCOVA 设置
    ANGLES   = ["HFSHK", "FFHF"]     # 两条角度都跑
    AXIS     = "Y"
    COND_A   = "WW-N-0_G"            # FM（基准）
    COND_B   = "WOW-N-nR_G"          # SM
    COV_COND = "B-0-0_G"             # 协变量（BF；trial-mean后再时间均值）
    ALPHA    = 0.05
    SAVE_DIR = "/Users/wangtingyu/Desktop/LMB/output/spm1d_ancova"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 可读名（只用于标题显示）
    pretty = {
        "WW-N-0_G":   "FM",
        "WOW-N-nR_G": "SM",
        "B-0-0_G":    "BF",
    }

    # 3) 逐角度执行 ANCOVA（方向= B-A：t>0 表示 SM>FM）
    for angle in ANGLES:
        title_txt = f"Comparison of {angle} angles between {pretty.get(COND_A,'FM')} and {pretty.get(COND_B,'SM')} markers"
        out = ancova_spm1d_two_groups(
            data_dict=data_dict,
            angle=angle,
            axis=AXIS,
            condA=COND_A,
            condB=COND_B,
            cov_cond=COV_COND,
            alpha=ALPHA,
            save_dir=SAVE_DIR,
            title=title_txt,
            figsize=(8, 4),         # 与 Mean±SD 图一致
            title_fs=15, label_fs=12, tick_fs=10,
            dpi=300,
            direction="A-B"         # 关键：有符号对比，阴影与方向一致
        )
        print(f"[saved] → {out['out_path']}")


def spm1d_paired_ttest_two_groups(
    data_dict,
    angle="HFSHK",
    axis="Y",
    condA="WW-N-0_G",           # 组A（与 condB 成对）
    condB="WOW-N-nR_G",         # 组B
    alpha=0.05,
    save_dir="/Users/wangtingyu/Desktop/LMB/output/spm1d_paired",
    title=None,
    figsize=(8,4),
    title_fs=13, label_fs=12, tick_fs=10,
    dpi=300,
    direction="B-A"             # "B-A": t>0 表示 B>A；"A-B": t>0 表示 A>B
):
    import os, numpy as np, matplotlib.pyplot as plt, spm1d
    os.makedirs(save_dir, exist_ok=True)

    # —— 工具：安全端点、P 文本、峰值标注 —— #
    def _int_endpoints(cl, n):
        e = getattr(cl, "endpoints", None)
        if e is not None:
            i0, i1 = int(e[0]), int(e[1])
            i0 = max(0, min(i0, n-1))
            i1 = max(i0+1, min(int(i1), n))
            return i0, i1
        idx = getattr(cl, "indices", getattr(cl, "I", None))
        if idx is None:
            return 0, 0
        idx = np.asarray(idx).ravel()
        if idx.size == 0:
            return 0, 0
        i0 = int(idx.min())
        i1 = int(idx.max()) + 1
        i0 = max(0, min(i0, n-1))
        i1 = max(i0+1, min(i1, n))
        return i0, i1

    def _fmt_P(p):
        return r"$P<0.001$" if (p is not None and p < 0.001) else rf"$P={p:.3f}$"

    def _annotate_peak(ax, x, y, text, prefer_above=True):
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        ax.set_ylim(ymin - 0.06*yspan, ymax + 0.18*yspan)
        ymin2, ymax2 = ax.get_ylim()
        yspan2 = ymax2 - ymin2
        offset = 0.06 * yspan2
        ytxt = y + (offset if prefer_above else -offset)
        ytxt = min(max(ytxt, ymin2 + 0.04*yspan2), ymax2 - 0.04*yspan2)
        va = "bottom" if ytxt >= y else "top"
        ax.annotate(
            text, xy=(x, y), xytext=(x, ytxt),
            ha="center", va=va, fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="0.5")
        )

    # —— 聚合（每被试两组 trial-mean 波形；只纳入两组都齐的被试） —— #
    subjects_ok, YA_list, YB_list = [], [], []
    L = None
    for subj in sorted(data_dict.keys()):
        nodeA = data_dict.get(subj, {}).get(condA, {}).get(angle, {}).get(axis)
        nodeB = data_dict.get(subj, {}).get(condB, {}).get(angle, {}).get(axis)
        if not (nodeA and nodeB):
            continue
        if L is None: 
            L = len(nodeA[0])
        if len(nodeB[0]) != L:
            continue

        YA = np.mean(np.vstack(nodeA), axis=0)  # trial-mean
        YB = np.mean(np.vstack(nodeB), axis=0)

        subjects_ok.append(subj)
        YA_list.append(YA)
        YB_list.append(YB)

    if not subjects_ok:
        raise RuntimeError("没有找到同时具备 condA/condB 的被试。")

    YA_mat = np.vstack(YA_list)   # (n_subj, L)
    YB_mat = np.vstack(YB_list)   # (n_subj, L)

    # —— 配对 t 检验（方向：B-A 表示 t>0 即 B>A；A-B 则相反） —— #
    if direction.upper() == "B-A":
        ti = spm1d.stats.ttest_paired(YB_mat, YA_mat).inference(alpha=alpha, two_tailed=True)
        comp = f"{condB} vs {condA}"
    else:  # "A-B"
        ti = spm1d.stats.ttest_paired(YA_mat, YB_mat).inference(alpha=alpha, two_tailed=True)
        comp = f"{condA} vs {condB}"

    # —— 作图 —— #
    fig, ax = plt.subplots(figsize=figsize)
    ti.plot(ax=ax)
    ti.plot_threshold_label(ax=ax)

    n_time = YA_mat.shape[1]
    ax.set_xlim(0, n_time-1); ax.margins(x=0)
    xt = np.linspace(0, n_time-1, 6)
    ax.set_xticks(xt); ax.set_xticklabels([f"{p:.0f}" for p in np.linspace(0, 100, 6)])

    if title is None:
        title = f"Paired t-test of {angle}: {comp}"
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("Stance phase (%)", fontsize=label_fs)
    ax.set_ylabel("SPM{t}", fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)

    # —— 峰值标注 —— #
    zt = np.asarray(ti.z).ravel()
    clusters = getattr(ti, "clusters", [])
    if clusters:
        for cl in clusters:
            i0, i1 = _int_endpoints(cl, zt.size)
            seg = zt[i0:i1]
            j_rel = int(np.argmax(np.abs(seg)))
            j = i0 + j_rel
            zpk = float(seg[j_rel])
            Ptxt = _fmt_P(cl.P)
            _annotate_peak(ax, j, zpk, f"t={zpk:.2f}\n{Ptxt}", prefer_above=(zpk>=0))
    else:
        j = int(np.argmax(np.abs(zt))); zpk = float(zt[j])
        _annotate_peak(ax, j, zpk, "n.s.", prefer_above=(zpk>=0))

    fig.tight_layout()
    fname = f"PairedT_{angle}-{axis}_{comp}.tiff".replace("/", "-").replace(" ", "")
    outp = os.path.join(save_dir, fname)
    fig.savefig(outp, dpi=dpi, format="tiff", bbox_inches="tight")
    plt.close(fig)

    # —— 控制台信息 —— 
    print(f"[Paired t] subjects={len(subjects_ok)}  L={L}")
    print(f"  saved: {outp}")

    # —— 返回 —— #
    return dict(
        spmi=ti,
        subjects=subjects_ok,
        out_path=outp
    )

'''
if __name__ == "__main__":
 
    print("spm1d version:", getattr(spm1d, "__version__", "unknown"))

    # 你的 data_dict 构建函数
    data_dict = build_full_data_dict(ROOT_PATH, TARGET_FILENAME, SUBJECTS)

    ANGLES   = ["HFSHK", "FFHF"]
    AXIS     = "Y"
    COND_A   = "WW-N-0_G"
    COND_B   = "WOW-N-nR_G"
    ALPHA    = 0.05
    SAVE_DIR = "/Users/wangtingyu/Desktop/LMB/output/spm1d_paired"
    os.makedirs(SAVE_DIR, exist_ok=True)

    pretty = {"WW-N-0_G":"FM", "WOW-N-nR_G":"SM"}

    for angle in ANGLES:
        title_txt = f"Comparision of {angle} angles between {pretty.get(COND_B,'B')}Ms and {pretty.get(COND_A,'A')}Ms "
        out = spm1d_paired_ttest_two_groups(
            data_dict=data_dict,
            angle=angle,
            axis=AXIS,
            condA=COND_A,
            condB=COND_B,
            alpha=ALPHA,
            save_dir=SAVE_DIR,
            title=title_txt,
            figsize=(8,4),
            title_fs=15, label_fs=12, tick_fs=10,
            dpi=300,
            direction="A-B"  # 令 t>0 表示 B>A（调换为 "A-B" 则 t>0 表示 A>B）
        )
        print(f"[saved] → {out['out_path']}")


    

if __name__ == "__main__":
    print("spm1d version:", getattr(spm1d, "__version__", "unknown"))

    # 1) 读取
    data_dict = build_full_data_dict(ROOT_PATH, TARGET_FILENAME, SUBJECTS)

    # 2) SPM1D 设置
    CONDITIONS = ["WOW-L-R_G", "WOW-L-nR_G", "WW-L-0_G"]  # SM_R, SM, FM
    ANGLES     = ["HFSHK", "FFHF"]
    AXES       = ["Y"]
    ALPHA      = 0.05
    USE_TRIAL_LEVEL_RM = True

    # 条件到可读名（用于标题）
    COND_LABEL = {
        "WOW-L-R_G":  "SMMs_R",
        "WOW-L-nR_G": "SMMs",
        "WW-L-0_G":   "FMMs",
    }

    # 主效应标题
    RM_TITLE = {
        ("HFSHK","Y"): "Comparison of HFSHK angles between FMMs, SMMs, and SMMs_R",
        ("FFHF","Y"):  "Comparison of FFHF angles between FMMs, SMMs, and SMMs_R",
    }

    # 输出与样式
    SAVE_DIR = "/Users/wangtingyu/Desktop/LMB/output/spm1d"
    os.makedirs(SAVE_DIR, exist_ok=True)
    TITLE_FONTSIZE = 15
    AXIS_FONTSIZE  = 12
    TICK_FONTSIZE  = 10
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize":  AXIS_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
    })

    
    def _format_P(p):
        if p is None:
            return r"$\mathit{P}$=n/a"
        return r"$\mathit{P}$<0.001" if p < 0.001 else rf"$\mathit{{P}}$={p:.3f}"

    def _annotate_peak(ax, x, y, text, prefer_above=True):
        # 边界保护：加上下余量再夹紧，避免顶到标题或边框
        ymin, ymax = ax.get_ylim()
        yspan = ymax - ymin
        ax.set_ylim(ymin - 0.06*yspan, ymax + 0.18*yspan)
        ymin2, ymax2 = ax.get_ylim()
        yspan2 = ymax2 - ymin2
        # 文本相对峰值的偏移
        ytxt = y + (0.06*yspan2 if prefer_above else -0.06*yspan2)
        ytxt = min(max(ytxt, ymin2 + 0.04*yspan2), ymax2 - 0.04*yspan2)
        va = "bottom" if ytxt >= y else "top"
        ax.annotate(
            text, xy=(x, y), xytext=(x, ytxt),
            ha="center", va=va, fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="0.5"),
        )

    # 3) 主循环
    for angle in ANGLES:
        for axis in AXES:
            banner = f"ANOVA | {angle}-{axis} | groups={CONDITIONS}"
            print("\n" + "="*len(banner)); print(banner); print("="*len(banner))

            # 3.1 trial 级长表 + 事后比较的“被试均值”
            try:
                YL, group, subjects, subj_order, cond_means = build_rm_longform(
                    data_dict, angle=angle, axis=axis,
                    cond_list=CONDITIONS, use_trial_level=USE_TRIAL_LEVEL_RM
                )
            except RuntimeError as e:
                print(f"[skip] {angle}-{axis}: {e}")
                continue

            print(f"进入 RM-ANOVA 的被试数: {len(subj_order)} / 总受试者数: {len(data_dict)}")
            if not USE_TRIAL_LEVEL_RM:
                print("注意：使用每被试每条件的均值（近似残差）；建议 USE_TRIAL_LEVEL_RM=True 更稳健。")

            # 3.2 主效应（F）：绘图 + 保存
            rm_title = RM_TITLE.get((angle, axis), f"SPM{{F}} RM-ANOVA | {angle}-{axis}")
            spmi_F, figF, axF = rm_anova_spm1d_compat(
                YL, group, subjects, CONDITIONS, alpha=ALPHA,
                title=rm_title, figsize=FIGSIZE_MSD,
                xlabel="Stance phase (%)", ylabel="SPM{F}",
                title_fs=TITLE_FONTSIZE, label_fs=AXIS_FONTSIZE, tick_fs=TICK_FONTSIZE
            )
            # 注意：rm_anova_spm1d_compat 内部已负责“每个显著簇的峰值标注”，
            # 这里不再重复标注，避免重复文字。
            outF = os.path.join(SAVE_DIR, f"RM_{angle}-{axis}.tiff")
            figF.savefig(outF, dpi=300, format="tiff", bbox_inches="tight")
            plt.close(figF)
            report_f_clusters(spmi_F, label=f"{angle}-{axis} (RM)")

            # 3.3 事后比较（t）：逐张图，标题含“of ANGLE”，阈值内峰值标注 + 保存
            results = posthoc_pairwise_rm_means(
                cond_means, CONDITIONS, alpha=ALPHA, two_tailed=True,
                plot=False, save_dir=None, angle=angle, axis=axis
            )

            for pair_name, spmij, minp in results:
                # pair_name 形如 "CJ - CI"（带空格）；解析代码并映射友好名
                parts = re.split(r"\s-\s", pair_name.strip())
                if len(parts) == 2:
                    cj_code, ci_code = parts[0].strip(), parts[1].strip()
                else:
                    toks = pair_name.replace("  ", " ").split("-")
                    cj_code, ci_code = toks[0].strip(), toks[-1].strip()
                cj_name = COND_LABEL.get(cj_code, cj_code)
                ci_name = COND_LABEL.get(ci_code, ci_code)
                a_name, b_name = sorted([ci_name, cj_name])  
                nice_title = f"Post-hoc tests between {a_name} and {b_name}"

                figT, axT = plt.subplots(figsize=(6,4))
                spmij.plot(ax=axT)
                spmij.plot_threshold_label(ax=axT)

                
                zt = np.asarray(spmij.z).ravel()
                n_time_T = zt.size
                axT.set_xlim(0, n_time_T - 1)
                axT.margins(x=0)
                xt = np.linspace(0, n_time_T - 1, 6)
                axT.set_xticks(xt)
                axT.set_xticklabels([f"{p:.0f}" for p in np.linspace(0, 100, 6)])

                axT.set_xlabel("Stance phase (%)", fontsize=AXIS_FONTSIZE)
                axT.set_ylabel("SPM{t}", fontsize=AXIS_FONTSIZE)
                axT.set_title(nice_title)

                
                thr = float(getattr(spmij, 'zstar', np.nan))
                clustersT = getattr(spmij, "clusters", [])
                if clustersT:
                    for cl in clustersT:
                        i0, i1 = _int_endpoints(cl, zt.size)  # 你前面定义的全局函数
                        seg = zt[i0:i1]
                        eps = 1e-12
                        if np.mean(seg) >= 0:
                            idx = np.where(seg >=  thr - eps)[0]
                            j_rel = idx[np.argmax(seg[idx])] if idx.size else int(np.argmax(seg))
                        else:
                            idx = np.where(seg <= -thr + eps)[0]
                            j_rel = idx[np.argmin(seg[idx])] if idx.size else int(np.argmin(seg))
                        j = i0 + j_rel
                        zpk = float(seg[j_rel])
                        _annotate_peak(axT, j, zpk, f"t={zpk:.2f}\n{_format_P(cl.P)}", prefer_above=(zpk>=0))
                else:
                    j = int(np.argmax(np.abs(zt)))
                    zpk = float(zt[j])
                    _annotate_peak(axT, j, zpk, f"t={zpk:.2f}\n(n.s.)", prefer_above=(zpk>=0))

                figT.tight_layout()
                outT = os.path.join(
                    SAVE_DIR,
                    f"POST_{angle}-{axis}_{cj_code}_vs_{ci_code}.tiff".replace("/", "-")
                )
                figT.savefig(outT, dpi=300, format="tiff", bbox_inches="tight")
                plt.close(figT)

                report_t_clusters(spmij, label=f"{angle}-{axis} posthoc {pair_name}")

            print(f"[done] {angle}-{axis} → 图像保存至：{SAVE_DIR}")

'''
if __name__ == "__main__":

    data_dict = build_full_data_dict(ROOT_PATH, TARGET_FILENAME, SUBJECTS)
    colors = {
        "WOW-L-R_G":  "#2CA02C",  
        "WOW-L-nR_G": "#1874CD",
        "WW-L-0_G":   "#FF0000",
    }
    labels = {
        "WOW-L-R_G":  "SMMs_R",   
        "WOW-L-nR_G": "SMMs",
        "WW-L-0_G":   "FMMs",
    }

    stats = plot_gait_cond_mean_sd(
        data_dict=data_dict,              
        angle_type="HFSHK",               # 角度类型：如 "HFSHK" / "FFHF"
        conditions=["WW-L-0_G", "WOW-L-nR_G","WOW-L-R_G"],  
        axis="Y",                         
        title="Angular change patterns of the HFSHK joint",
        xlabel="Stance phase (%)",
        ylabel="HFSHK in/eversion angle (°)",
        color_map=colors,                 
        legend_labels=labels,             
        legend_ncol=1,                    
        legend_loc="best",                
        include_n=True,                   
        line_kw={"linewidth": 2},         
        fill_alpha=0.18,                  
        figsize=(8,4),                    
        save_path="/Users/wangtingyu/Desktop/LMB/output/Shoe_mounted/HFSHK_mean_LWI.tiff", 
        save_dpi=300                      
    )
