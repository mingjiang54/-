# -*- coding: utf-8 -*-
# MK for HL(t) = p4 + p5 (4、5级热岛面积占比)
# Includes: MK(含并列修正) + Theil–Sen + UF/UB + TFPW + Pettitt
# 输出斜率：比例/年 以及 百分比点/年（pp/年）
# 仅依赖：NumPy

import math
import numpy as np

# ========= 参数与输入（按需改动） =========
years = np.array([2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023], float)

# HL 可用 0–1（比例）或 0–100（百分比）。下面是示例，请替换为你的数据：
HL = np.array([37.2, 35.8, 35.3, 33.8, 36.8, 35.6, 35.2, 38, 37.9, 33.3, 33], float)

alpha     = 0.05          # 显著性水平（可设 0.05）
direction = 'up'          # 单侧方向：'up' 或 'down'
delta_pp  = 0.30          # 实用显著性阈值：pp/年（例如 0.30 pp/年 = 3 pp/十年）

# ========= 预处理：统一单位 =========
# 自动识别输入是否为百分比：若均值 > 1.5 视为百分比，转为 0–1 比例以便计算
is_percent_input = np.nanmean(HL) > 1.5
HL_frac = HL / 100.0 if is_percent_input else HL  # 统一比例
unit_note = "(输入为百分比，已自动换算为比例)" if is_percent_input else "(输入为比例)"
delta_frac = delta_pp / 100.0                     # 实用阈值换算为比例/年

# 缺失剔除
mask = (~np.isnan(years)) & (~np.isnan(HL_frac))
years  = years[mask]
HL_frac = HL_frac[mask]
n = len(HL_frac)
assert n >= 3, "序列太短，MK 至少需要 3 个点"

# ========= 工具函数 =========
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def z_from_alpha_two_sided(alpha):
    return 1.96 if abs(alpha-0.05) < 1e-9 else (1.645 if abs(alpha-0.10) < 1e-9 else 1.96)

def mk_test_with_ties(x):
    x = np.asarray(x, float)
    n = len(x)
    S = 0
    for k in range(n-1):
        S += np.sign(x[k+1:] - x[k]).sum()

    # 并列修正
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2*counts + 5))
    varS = (n*(n-1)*(2*n+5) - tie_term) / 18.0

    if S > 0:
        Z = (S - 1) / math.sqrt(varS)
    elif S < 0:
        Z = (S + 1) / math.sqrt(varS)
    else:
        Z = 0.0

    p_two = 2.0 * (1.0 - norm_cdf(abs(Z)))
    tau = S / (0.5 * n * (n - 1))
    return S, varS, Z, p_two, tau

def theil_sen_slope(x, t):
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    n = len(x)
    slopes = []
    for i in range(n-1):
        dt = t[i+1:] - t[i]
        dy = x[i+1:] - x[i]
        s = dy[dt != 0] / dt[dt != 0]
        slopes.extend(s.tolist())
    slopes = np.array(slopes, float)
    slope = np.median(slopes)
    intercept = np.median(x - slope * t)

    # 近似95%CI（基于 Var(S) 的秩位置）
    S, varS, _, _, _ = mk_test_with_ties(x)
    M = n*(n-1)//2
    z = z_from_alpha_two_sided(0.05)
    C = z * math.sqrt(varS)
    k_low  = int(max(0, min(M-1, math.floor((M - C) / 2.0))))
    k_high = int(max(0, min(M-1, math.ceil( (M + C) / 2.0 ))))
    s_sorted = np.sort(slopes)
    ciL, ciU = s_sorted[k_low], s_sorted[k_high]
    return slope, intercept, (ciL, ciU)

def sequential_mk_UF_UB(x):
    x = np.asarray(x, float)
    n = len(x)
    Sk = np.zeros(n)
    for k in range(1, n):
        Sk[k] = Sk[k-1] + np.sign(x[k] - x[:k]).sum()
    E = np.array([i*(i-1)/4.0 for i in range(n)], float)
    V = np.array([i*(i-1)*(2*i+5)/72.0 for i in range(n)], float)
    V[0:2] = 1.0
    UF = (Sk - E) / np.sqrt(V)

    xr = x[::-1]
    Skr = np.zeros(n)
    for k in range(1, n):
        Skr[k] = Skr[k-1] + np.sign(xr[k] - xr[:k]).sum()
    UBr = (Skr - E) / np.sqrt(V)
    UB  = -UBr[::-1]
    return UF, UB

def detect_change_points(UF, UB, years, alpha=0.05):
    years = np.asarray(years, int)
    zthr = z_from_alpha_two_sided(alpha)
    diff = UF - UB
    cand = []
    for k in range(1, len(UF)):
        if (diff[k-1] * diff[k] <= 0) and (abs(UF[k]) >= zthr or abs(UB[k]) >= zthr):
            cand.append(int(years[k]))
    return sorted(set(cand))

def yue_pilon_tfpw_mk(x, t):
    beta, b0, _ = theil_sen_slope(x, t)
    r = x - (beta * t + b0)
    rho = np.corrcoef(r[:-1], r[1:])[0,1] if r.std() > 0 else 0.0
    if np.isnan(rho):
        rho = 0.0
    rp  = r[1:] - rho * r[:-1]
    xpw = rp + (beta * t[1:] + b0)   # 长度 n-1
    return mk_test_with_ties(xpw), rho

def pettitt_test(x):
    x = np.asarray(x, float); n = len(x)
    Kmax = 0; t_star = None
    for t in range(n):
        s = 0
        for i in range(n):
            for j in range(n):
                if i <= t and j > t:
                    s += np.sign(x[i] - x[j])
        K = abs(s)
        if K > Kmax:
            Kmax = K; t_star = t
    p = 2 * math.exp((-6 * (Kmax ** 2)) / (n ** 3 + n ** 2))
    p = min(1.0, p)  # 近似公式可能 >1，截断
    return t_star, Kmax, p

# ========= 计算 =========
S, varS, Z, p_two, tau = mk_test_with_ties(HL_frac)
slope, intercept, (ciL, ciU) = theil_sen_slope(HL_frac, years)
UF, UB = sequential_mk_UF_UB(HL_frac)
candidates = detect_change_points(UF, UB, years, alpha=alpha)
(mk_pw_S, mk_pw_varS, mk_pw_Z, mk_pw_p2, mk_pw_tau), rho = yue_pilon_tfpw_mk(HL_frac, years)
t_star, Kmax, p_pettitt = pettitt_test(HL_frac)
year_star = int(years[t_star]) if t_star is not None else None

# 单侧 p
if direction == 'up':
    p_one = 1.0 - norm_cdf(Z)
elif direction == 'down':
    p_one = norm_cdf(Z)
else:
    raise ValueError("direction 需为 'up' 或 'down'")

# 实用显著性：CI 完全落在 (-delta, +delta) 内视为“无具有实际意义的趋势”
equiv_small = (ciL > -delta_frac) and (ciU < delta_frac)

# 辅助：把斜率换算为“百分比点/年”
def to_pp_per_year(x):
    return x * 100.0

# ========= 输出 =========
print("=== 输入口径 ===")
print(f"HL 序列单位：{'百分比(0-100)' if is_percent_input else '比例(0-1)'}  {unit_note}")
print(f"样本数 n = {n}\n")

print("=== Mann–Kendall 趋势检验（含并列修正）===")
print(f"S = {S:.0f}, Var(S) = {varS:.3f}, Z = {Z:.3f}, p(双侧) = {p_two:.4f}, tau = {tau:.3f}")
print(f"趋势判定（双侧, α={alpha:.2f}）: {'显著' if p_two < alpha else '不显著'}")

print("\n=== 单侧显著性 ===")
print(f"方向: {'上升' if direction=='up' else '下降'}, α={alpha:.2f}, p(单侧) = {p_one:.4f} -> "
      f"{'显著' if p_one < alpha else '不显著'}")

print("\n=== Theil–Sen 斜率（幅度）===")
print(f"β = {slope:.6f} /年（比例/年）  ≈ {to_pp_per_year(slope):.4f} pp/年")
print(f"95%CI: [{ciL:.6f}, {ciU:.6f}] /年（比例/年）  ≈ [{to_pp_per_year(ciL):.4f}, {to_pp_per_year(ciU):.4f}] pp/年")
print(f"实用阈值 delta = {delta_pp:.3f} pp/年  ->  "
      f"{'无具有实际意义的趋势（|β|<delta 且 CI 完全落入）' if equiv_small else '无法判定为“足够小”的趋势'}")
print(f"截距(中位) = {intercept:.6f}  （用于 y = β*t + 截距；如需可对年份中心化）")

print("\n=== 顺序 MK 突变检验（UF/UB）===")
zthr = z_from_alpha_two_sided(alpha)
print("UF:", np.round(UF, 3))
print("UB:", np.round(UB, 3))
print(f"显著带阈值 ±{zthr:.3f}（联动 α={alpha:.2f}）")
print("候选突变年（探索性）:", candidates if len(candidates)>0 else "未检出")

print("\n=== 自相关修正 MK（TFPW）===")
print(f"rho_AR1(residuals) = {rho:.3f}")
print(f"Z = {mk_pw_Z:.3f}, p(双侧) = {mk_pw_p2:.4f}, tau = {mk_pw_tau:.3f}  "
      f"-> {'显著' if mk_pw_p2 < alpha else '不显著'} (α={alpha:.2f})")

print("\n=== Pettitt 变点检验 ===")
print(f"候选变点年份 = {year_star}, K = {Kmax:.0f}, p = {p_pettitt:.3f}  "
      f"-> {'显著' if p_pettitt < alpha else '不显著'} (α={alpha:.2f})")
