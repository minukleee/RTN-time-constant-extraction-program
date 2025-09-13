import os
import numpy as np

# ====== 출력 폴더 설정 ======
OUTDIR = "rtn_out"   # 원하는 이름으로 변경 가능

# ====== 스무딩/양자화 설정 ======
TOP_CHANGE_FRAC     = 0.999
SMOOTH_WIN_SAMPLES  = 11
EDGE_PAD_SAMPLES    = 2

CORE_MARGIN_SAMPLES = 8
LOW_TRIM  = (0.05, 0.50)
HIGH_TRIM = (0.50, 0.95)
REFINE_CENTER_ONCE  = True
LEVEL_FROM_RAW      = True

# ====== 히스토그램(기본) 설정 ======
#  "fd" | "sqrt" | 정수(Nbins) | ("width", 고정 bin 폭[초])
HIST_MODE        = ("width", 0.010)
HIST_USE_SHARED  = True
HIST_X_RANGE     = None

# ====== 추가: 50-bin 고정 히스토그램 ======
NBINS_FIXED      = 50
REC_DURATION_MAX = 1.0   # 측정 0~1 s 가정

MIN_T_POS        = 0.0   # 0 이하 dwell 제거

# ---------- 유틸 ----------
def outpath(name):
    return os.path.join(OUTDIR, name)

def read_txt(path):
    arr = np.loadtxt(path, dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1)

def moving_average_same(x, win):
    if win is None or win <= 1:
        return x
    k = int(win)
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, kernel, mode="same")

def edge_preserving_smooth(data_zero_mean):
    n = data_zero_mean.size
    if n == 0:
        return data_zero_mean.copy()
    d = np.abs(np.diff(data_zero_mean))
    thr = float(np.quantile(d, TOP_CHANGE_FRAC)) if d.size else float("inf")
    edge_idx = np.where(d >= thr)[0]
    is_edge = np.zeros(n, dtype=bool)
    for i in edge_idx:
        i0 = max(0, i - EDGE_PAD_SAMPLES)
        i1 = min(n - 1, i + 1 + EDGE_PAD_SAMPLES)
        is_edge[i0:i1+1] = True
    xs_full = moving_average_same(data_zero_mean, SMOOTH_WIN_SAMPLES)
    xs = xs_full.copy()
    xs[is_edge] = data_zero_mean[is_edge]
    return xs

def _trimmed_mean(vals, qlo, qhi):
    if vals.size == 0:
        return float("nan")
    lo = float(np.quantile(vals, qlo))
    hi = float(np.quantile(vals, qhi))
    core = vals[(vals >= lo) & (vals <= hi)]
    if core.size == 0:
        return float(np.mean(vals))
    return float(np.mean(core))

def _state_from_center(x, center):
    return (x >= center).astype(np.int8)  # 1=High, 0=Low

def _exclude_boundaries(state, margin, length):
    cp = np.where(np.diff(state) != 0)[0]
    keep = np.ones(length, dtype=bool)
    for i in cp:
        a = max(0, i - margin + 1)
        b = min(length - 1, i + margin)
        keep[a:b+1] = False
    return keep

def quantize_classify_plateau(xs_zero_mean, data_zero_mean):
    p10 = float(np.quantile(xs_zero_mean, 0.10))
    p99 = float(np.quantile(xs_zero_mean, 0.99))
    center = 0.5 * (p10 + p99)

    n = xs_zero_mean.size
    state = _state_from_center(xs_zero_mean, center)

    keep = _exclude_boundaries(state, CORE_MARGIN_SAMPLES, n)
    low_core  = (state == 0) & keep
    high_core = (state == 1) & keep

    level_base = data_zero_mean if LEVEL_FROM_RAW else xs_zero_mean
    low_level  = _trimmed_mean(level_base[low_core],  LOW_TRIM[0],  LOW_TRIM[1])
    high_level = _trimmed_mean(level_base[high_core], HIGH_TRIM[0], HIGH_TRIM[1])

    if REFINE_CENTER_ONCE and np.isfinite(low_level) and np.isfinite(high_level):
        new_center = 0.5 * (low_level + high_level)
        if new_center != center:
            center = new_center
            state = _state_from_center(xs_zero_mean, center)
            keep = _exclude_boundaries(state, CORE_MARGIN_SAMPLES, n)
            low_core  = (state == 0) & keep
            high_core = (state == 1) & keep
            low_level  = _trimmed_mean(level_base[low_core],  LOW_TRIM[0],  LOW_TRIM[1])
            high_level = _trimmed_mean(level_base[high_core], HIGH_TRIM[0], HIGH_TRIM[1])

    q = np.where(xs_zero_mean < center, low_level, high_level)
    return q, low_level, high_level, center

def dwell_times_all(time, q, center):
    state = (q >= center).astype(np.int8)
    n = state.size
    if n == 0:
        return np.array([]), np.array([])
    last_state = int(state[0])
    last_t = float(time[0])
    tau_low, tau_high = [], []
    for i in range(1, n):
        if state[i] != last_state:
            dt = float(time[i] - last_t)
            if last_state == 0:
                tau_low.append(dt)
            else:
                tau_high.append(dt)
            last_state = int(state[i])
            last_t = float(time[i])
    tL = np.asarray(tau_low,  float)
    tH = np.asarray(tau_high, float)
    tL = tL[np.isfinite(tL) & (tL > MIN_T_POS)]
    tH = tH[np.isfinite(tH) & (tH > MIN_T_POS)]
    return tL, tH

def dwell_means(time, q, center):
    state = (q >= center).astype(np.int8)
    n = state.size
    if n == 0:
        return float("nan"), float("nan")
    last_state = int(state[0])
    last_t = float(time[0])
    tau_e, tau_c = [], []
    for i in range(1, n):
        if state[i] != last_state:
            dt = float(time[i] - last_t)
            if last_state == 0:
                tau_e.append(dt)
            else:
                tau_c.append(dt)
            last_state = int(state[i])
            last_t = float(time[i])
    m_e = float(np.mean(tau_e)) if len(tau_e) else float("nan")
    m_c = float(np.mean(tau_c)) if len(tau_c) else float("nan")
    return m_e, m_c

# ----- 유연형 히스토그램(기존) -----
def _build_edges_from_mode(t_all, mode, x_range=None):
    t = np.asarray(t_all, float)
    t = t[np.isfinite(t) & (t > 0)]
    if x_range is not None and t.size > 0:
        t = t[(t >= float(x_range[0])) & (t <= float(x_range[1]))]
    if t.size == 0:
        return None
    if isinstance(mode, tuple) and len(mode) == 2 and str(mode[0]).lower() == "width":
        bw = float(mode[1])
        tmin = float(np.min(t)) if x_range is None else float(x_range[0])
        tmax = float(np.max(t)) if x_range is None else float(x_range[1])
        if tmax <= tmin:
            tmax = tmin + bw
        nb  = int(np.ceil((tmax - tmin) / bw))
        edges = tmin + bw * np.arange(nb + 1)
        return edges
    if isinstance(mode, int):
        nb = int(mode)
        tmin = float(np.min(t)) if x_range is None else float(x_range[0])
        tmax = float(np.max(t)) if x_range is None else float(x_range[1])
        if tmax <= tmin:
            tmax = tmin + (np.max(t) - np.min(t) if t.size > 1 else 1.0)
        return np.linspace(tmin, tmax, nb + 1)
    if str(mode).lower() == "sqrt":
        nb = max(5, int(np.sqrt(max(1, t.size))))
        return np.linspace(float(np.min(t)), float(np.max(t)), nb + 1)
    # Freedman–Diaconis
    q75, q25 = np.percentile(t, [75, 25])
    iqr = q75 - q25
    h = 2.0 * iqr * (t.size ** (-1.0 / 3.0))
    if h <= 0:
        h = 1.06 * np.std(t) * (t.size ** (-1.0 / 5.0))
        if h <= 0:
            h = (np.max(t) - np.min(t)) / max(1, int(np.sqrt(t.size)))
    nb = int(np.ceil((np.max(t) - np.min(t)) / h)) if h > 0 else int(np.sqrt(t.size))
    nb = max(5, nb)
    return np.linspace(float(np.min(t)), float(np.max(t)), nb + 1)

def _hist_from_edges(t, edges, x_range=None):
    t = np.asarray(t, float)
    t = t[np.isfinite(t) & (t > 0)]
    if x_range is not None:
        t = t[(t >= float(x_range[0])) & (t <= float(x_range[1]))]
    if t.size == 0 or edges is None:
        return np.array([]), np.array([])
    counts, e = np.histogram(t, bins=edges)
    centers = 0.5 * (e[:-1] + e[1:])
    return centers, counts.astype(int)

def save_hist_txt(path, centers, counts):
    np.savetxt(path,
               np.column_stack([centers, counts]),
               fmt="%.11f %d",
               header="bin_center  count",
               comments="")

# ----- 50-bin 고정 -----
def fixed_nbins_hist(t, nbins, rec_duration_max):
    t = np.asarray(t, float)
    t = t[np.isfinite(t) & (t > 0)]
    if t.size == 0:
        edges = np.linspace(0.0, rec_duration_max, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts  = np.zeros_like(centers, dtype=int)
        return centers, counts
    xmax_data = float(np.max(t)) * 1.000001
    xmax = float(min(rec_duration_max, xmax_data))
    if xmax <= 0:
        xmax = rec_duration_max
    edges = np.linspace(0.0, xmax, nbins + 1)
    counts, e = np.histogram(t, bins=edges)
    centers = 0.5 * (e[:-1] + e[1:])
    return centers, counts.astype(int)

# ---------- 메인 ----------
def main():
    # 출력 폴더 생성
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)

    # 입력 로드
    time = read_txt("time.txt")
    data = read_txt("data.txt")
    if time.size != data.size:
        raise ValueError("Length mismatch: time={} != data={}".format(time.size, data.size))

    mean_val = float(np.mean(data))
    data0 = data - mean_val

    # 스무딩 → 양자화
    xs = edge_preserving_smooth(data0)
    q0, low_level, high_level, center = quantize_classify_plateau(xs, data0)

    # dwell times
    tau_low, tau_high = dwell_times_all(time, q0, center)
    mean_tau_e, mean_tau_c = dwell_means(time, q0, center)

    # 시계열 저장(평균 복원)
    xs_out = xs + mean_val
    q_out  = q0 + mean_val
    np.savetxt(outpath("data_smoothed.txt"),
               np.column_stack([time, xs_out]),
               fmt="%.11f",
               header="Time  Data_Smoothed",
               comments="")
    np.savetxt(outpath("quantized.txt"),
               np.column_stack([time, q_out]),
               fmt="%.11f",
               header="Time  Quantized_data",
               comments="")

    # 평균 체류시간
    with open(outpath("tau_e.txt"), "w", encoding="utf-8") as f:
        f.write("{:.11f}\n".format(mean_tau_e))
    with open(outpath("tau_c.txt"), "w", encoding="utf-8") as f:
        f.write("{:.11f}\n".format(mean_tau_c))

    # 원자료 리스트
    np.savetxt(outpath("tau_h_raw.txt"), tau_high, fmt="%.11f")
    np.savetxt(outpath("tau_l_raw.txt"), tau_low,  fmt="%.11f")

    # 기본 히스토그램(유연형)
    if HIST_USE_SHARED:
        all_for_edges = np.concatenate([tau_high, tau_low]) if tau_high.size + tau_low.size > 0 else np.array([0.0, 1.0])
        edges = _build_edges_from_mode(all_for_edges, HIST_MODE, x_range=HIST_X_RANGE)
        c_h, n_h = _hist_from_edges(tau_high, edges, x_range=HIST_X_RANGE)
        c_l, n_l = _hist_from_edges(tau_low,  edges, x_range=HIST_X_RANGE)
    else:
        edges_h = _build_edges_from_mode(tau_high, HIST_MODE, x_range=HIST_X_RANGE)
        edges_l = _build_edges_from_mode(tau_low,  HIST_MODE, x_range=HIST_X_RANGE)
        c_h, n_h = _hist_from_edges(tau_high, edges_h, x_range=HIST_X_RANGE)
        c_l, n_l = _hist_from_edges(tau_low,  edges_l, x_range=HIST_X_RANGE)
    save_hist_txt(outpath("tau_h.txt"), c_h, n_h)
    save_hist_txt(outpath("tau_l.txt"), c_l, n_l)

    # 50-bin 고정 히스토그램
    c_h50, n_h50 = fixed_nbins_hist(tau_high, NBINS_FIXED, REC_DURATION_MAX)
    c_l50, n_l50 = fixed_nbins_hist(tau_low,  NBINS_FIXED, REC_DURATION_MAX)
    save_hist_txt(outpath("tau_h_50.txt"), c_h50, n_h50)
    save_hist_txt(outpath("tau_l_50.txt"), c_l50, n_l50)

    print("Output folder: {}".format(os.path.abspath(OUTDIR)))
    print("Low={:.3e}, High={:.3e}, Center={:.3e}".format(low_level, high_level, center))
    print("Tau_E_mean={:.6g}s, Tau_C_mean={:.6g}s".format(mean_tau_e, mean_tau_c))
    print("Events  High={}, Low={}".format(tau_high.size, tau_low.size))
    print("Saved files: data_smoothed.txt, quantized.txt, tau_e.txt, tau_c.txt,")
    print("             tau_h_raw.txt, tau_l_raw.txt, tau_h.txt, tau_l.txt,")
    print("             tau_h_50.txt, tau_l_50.txt")

if __name__ == "__main__":
    main()
