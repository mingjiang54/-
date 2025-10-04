# -*- coding: utf-8 -*-
# MK（含并列修正）+ Theil–Sen + 顺序MK(UF/UB) + TFPW + Pettitt
# 支持：单侧/可配置alpha、UF/UB阈值联动、实用显著性delta评估
# 仅依赖标准库与 NumPy

import math
import numpy as np

# ========= 参数与输入（按需改动） =========
years  = np.array([2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], dtype=float)
# 示例：大区域夏季年均温（单位：°C）；改成你的序列即可
series = np.array([31.675, 31.917, 32.374, 33.107, 32.942, 32.197, 32.453, 33.175, 32.763, 32.635, 32.801  ], dtype=float)

alpha      = 0.10          # 显著性水平（0.10 或 0.05 常用）
direction  = 'up'          # 单侧方向：'up'（只检上升）或 'down'（只检下降）
delta      = 0.05          # 实用显著性阈值（°C/年），例：0.05 °C/年=0.5 °C/10年

# ========= 预处理 =========
mask   = (~np.isnan(years)) & (~np.isnan(series))
years  = years[mask]
series = series[mask]
n = len(series)
assert n >= 3, "序列太短，MK 至少需要 3 个点"

# ========= 工具函数 =========
def norm_cdf(z):
    """标准正态 CDF（用 erf 实现，避免 SciPy 依赖）"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def z_from_alpha_two_sided(alpha):
    # 常用alpha的近似临界值；其他alpha回退到1.96
    return 1.96 if abs(alpha-0.05) < 1e-9 else (1.645 if abs(alpha-0.10) < 1e-9 else 1.96)

def mk_test_with_ties(x):
    """
    Mann–Kendall 原始检验（含并列修正）
    返回：S, Var(S), Z, p(双侧), tau
    """
    x = np.asarray(x, float)
    n = len(x)
    S = 0
    for k in range(n-1):
        S += np.sign(x[k+1:] - x[k]).sum()

    # 并列修正
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
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
    """
    Theil–Sen 斜率（全部点对斜率的中位数）+ 近似95%CI（基于 Var(S) 的秩位置）
    返回：slope(°C/年), intercept, (ciL, ciU)
    """
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    n = len(x)
    slopes = []
    for i in range(n-1):
        dt = t[i+1:] - t[i]
        dy = x[i+1:] - x[i]
        valid = (dt != 0)
        s = dy[valid] / dt[valid]
        slopes.extend(s.tolist())
    slopes = np.array(slopes, float)
    slope = np.median(slopes)
    intercept = np.median(x - slope * t)

    # 近似 CI（Helsel & Hirsch; 用 Var(S) 推导秩位置）
    S, varS, _, _, _ = mk_test_with_ties(x)
    M = n * (n - 1) // 2
    z = z_from_alpha_two_sided(0.05)  # 95%CI
    C = z * math.sqrt(varS)
    k_low  = int(max(0, min(M - 1, math.floor((M - C) / 2.0))))
    k_high = int(max(0, min(M - 1, math.ceil((M + C) / 2.0))))
    s_sorted = np.sort(slopes)
    ciL, ciU = s_sorted[k_low], s_sorted[k_high]
    return slope, intercept, (ciL, ciU)

def sequential_mk_UF_UB(x):
    """
    顺序 MK（UF/UB），用于突变探索。
    说明：常见实现，未逐步处理并列，仅作探索性参考。
    """
    x = np.asarray(x, float)
    n = len(x)
    Sk = np.zeros(n)
    for k in range(1, n):
        Sk[k] = Sk[k-1] + np.sign(x[k] - x[:k]).sum()
    E = np.array([i*(i-1)/4.0 for i in range(n)], float)
    V = np.array([i*(i-1)*(2*i+5)/72.0 for i in range(n)], float)
    V[0:2] = 1.0  # 防止除0
    UF = (Sk - E) / np.sqrt(V)

    xr = x[::-1]
    Skr = np.zeros(n)
    for k in range(1, n):
        Skr[k] = Skr[k-1] + np.sign(xr[k] - xr[:k]).sum()
    UBr = (Skr - E) / np.sqrt(V)
    UB = -UBr[::-1]  # 反向对齐并取负
    return UF, UB

def detect_change_points(UF, UB, years, alpha=0.05):
    """
    基于 UF/UB 的交叉并越过阈值给出候选突变年（探索性）
    阈值随 alpha 联动：alpha=0.10 -> 1.645；alpha=0.05 -> 1.96
    """
    years = np.asarray(years, int)
    zthr = z_from_alpha_two_sided(alpha)
    diff = UF - UB
    cand = []
    for k in range(1, len(UF)):
        crossed = (diff[k-1] * diff[k] <= 0)
        beyond  = (abs(UF[k]) >= zthr) or (abs(UB[k]) >= zthr)
        if crossed and beyond:
            cand.append(int(years[k]))
    return sorted(set(cand))

def yue_pilon_tfpw_mk(x, t):
    """
    TFPW（Yue–Pilon）：先用 Sen 斜率去趋势 -> 估计 AR(1) -> 预白化残差 -> 加回趋势 -> MK
    返回：(S, VarS, Z, p_two, tau), rho_AR1
    """
    beta, b0, _ = theil_sen_slope(x, t)
    r = x - (beta * t + b0)                 # 残差
    rho = np.corrcoef(r[:-1], r[1:])[0,1] if r.std() > 0 else 0.0
    if np.isnan(rho):
        rho = 0.0
    rp  = r[1:] - rho * r[:-1]              # 预白化残差
    xpw = rp + (beta * t[1:] + b0)          # 加回趋势后的序列（长度 n-1）
    return mk_test_with_ties(xpw), rho

def pettitt_test(x):
    """
    Pettitt 单一变点非参数检验（O(n^2)实现）
    返回：t_star(索引)、Kmax、p
    """
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
    return t_star, Kmax, p

# ========= 计算 =========
S, varS, Z, p_two, tau = mk_test_with_ties(series)
slope, intercept, (ciL, ciU) = theil_sen_slope(series, years)
UF, UB = sequential_mk_UF_UB(series)
candidates = detect_change_points(UF, UB, years, alpha=alpha)
(mk_pw_S, mk_pw_varS, mk_pw_Z, mk_pw_p2, mk_pw_tau), rho = yue_pilon_tfpw_mk(series, years)
t_star, Kmax, p_pettitt = pettitt_test(series)
year_star = int(years[t_star]) if t_star is not None else None

# 单侧 p
if direction == 'up':
    p_one = 1.0 - norm_cdf(Z)
elif direction == 'down':
    p_one = norm_cdf(Z)
else:
    raise ValueError("direction 需为 'up' 或 'down'")

# 实用显著性（等效性）判断：CI 完全落在 (-delta, +delta) 内视为“无具有实际意义的趋势”
equiv_small_trend = (ciL > -delta) and (ciU < delta)

# ========= 输出 =========
print("=== Mann–Kendall 趋势检验（含并列修正）===")
print(f"n = {n}")
print(f"S = {S:.0f}, Var(S) = {varS:.3f}, Z = {Z:.3f}, p(双侧) = {p_two:.4f}")
print(f"Kendall's tau = {tau:.3f}")
sig_two = (p_two < alpha)
print(f"趋势判定（双侧, α={alpha:.2f}）: {'显著' if sig_two else '不显著'}")

print("\n=== 单侧显著性（方向: %s, α=%.2f）===" % ('上升' if direction=='up' else '下降', alpha))
print(f"p(单侧) = {p_one:.4f} -> {'显著' if p_one < alpha else '不显著'}")

print("\n=== Theil–Sen 斜率（幅度）===")
print(f"Sen slope = {slope:.5f} °C/年")
print(f"95% CI of slope ≈ [{ciL:.5f}, {ciU:.5f}] °C/年")
print(f"实用阈值 delta = {delta:.3f} °C/年 -> "
      f"{'无具有实际意义的趋势（|β|<delta）' if equiv_small_trend else '无法判定为“足够小”的趋势'}")
print(f"Median intercept = {intercept:.5f} （用于 y = {slope:.5f}*t + {intercept:.5f}；若需可改用年份中心化）")

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

print("\n=== Pettitt 突变检验 ===")
print(f"候选变点年份 = {year_star}, K = {Kmax:.0f}, p = {p_pettitt:.3f}  "
      f"-> {'显著' if p_pettitt < alpha else '不显著'} (α={alpha:.2f})")
