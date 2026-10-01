# 1走行の距離軌跡 R(t) を、横方向離隔 d（数字1個）に畳む。
#
# 主張(2)「ライン間の差 > 同一条件内のばらつき」を定量で判定するための本体。
# 曲線を目で見比べるより感度が高い。150フレームぶんの情報が d̂ 1個に集約されるため、
# フレームごとのばらつきが平均化される。
#
# モデル:
#   R(s)^2 = (s - X0)^2 + d^2        s: 走路に沿った位置[m]
#
# 2つの使い方があり、現地の条件で選ぶ:
#   fit_with_s        … 映像からコーン通過時刻が取れた場合。s(t) を与える（主張(1) の予測版）
#   fit_const_speed   … 夜間収録で映像が使えない場合。s = v t を仮定し v も同時に推定
#
# どちらも変数変換すると線形最小二乗になるので、初期値も反復も要らない。
# そのうえで R の残差に対する Gauss-Newton を数回回して仕上げる
# （R^2 での最小二乗は遠方を重く見るため、そのままだと遠方寄りの当てはめになる）。

import numpy as np

GN_ITER = 8        # Gauss-Newton の反復回数。10回も回せば動かなくなる


def _refine(r, s_of, params, jac):
    """R の残差に対する Gauss-Newton。params を更新して返す"""
    p = np.array(params, dtype=float)
    for _ in range(GN_ITER):
        model, j = s_of(p), jac(p)                      # (T,), (T, P)
        dp, *_ = np.linalg.lstsq(j, r - model, rcond=None)
        p = p + dp
        if np.max(np.abs(dp)) < 1e-9:
            break
    return p


def fit_with_s(r, s):
    """s(t) が既知の場合。戻り値 (X0, d, rms)

    r : 観測した斜距離 [m]  (T,)
    s : 走路に沿った位置 [m] (T,)

    R^2 - s^2 = -2 s X0 + (X0^2 + d^2) と書けるので (X0, X0^2+d^2) について線形。
    """
    r, s = np.asarray(r, float), np.asarray(s, float)
    a = np.stack([-2 * s, np.ones_like(s)], axis=1)      # (T, 2)
    (x0, b), *_ = np.linalg.lstsq(a, r**2 - s**2, rcond=None)
    d = np.sqrt(max(b - x0**2, 1e-9))

    def model(p):
        return np.hypot(s - p[0], p[1])

    def jac(p):
        m = model(p)
        return np.stack([-(s - p[0]) / m, p[1] / m], axis=1)   # (T, 2)

    x0, d = _refine(r, model, (x0, d), jac)
    return float(x0), float(abs(d)), float(np.sqrt(np.mean((r - model((x0, d)))**2)))


def fit_const_speed(r, t):
    """s(t) が取れない場合。等速を仮定して v も推定する。戻り値 (v, t_min, d, rms)

    r : 観測した斜距離 [m] (T,)
    t : 時刻 [s]          (T,)

    s = v t とすると R^2 = v^2 t^2 - 2 a v t + (a^2 + d^2)。
    (v^2, a v, a^2+d^2) について線形なので、まずそこで解く。
    t_min は最接近の時刻で、走路のどこを最接近点が通ったかに対応する。
    """
    r, t = np.asarray(r, float), np.asarray(t, float)
    a_mat = np.stack([t**2, -2 * t, np.ones_like(t)], axis=1)    # (T, 3)
    (p, q, c), *_ = np.linalg.lstsq(a_mat, r**2, rcond=None)
    v = np.sqrt(max(p, 1e-9))
    t_min = q / max(p, 1e-9)                                     # a v / v^2 = a / v
    d = np.sqrt(max(c - (t_min * v)**2, 1e-9))

    def model(pr):
        return np.hypot(pr[0] * (t - pr[1]), pr[2])

    def jac(pr):
        m, dt = model(pr), t - pr[1]
        return np.stack([pr[0] * dt**2 / m, -pr[0]**2 * dt / m, pr[2] / m], axis=1)   # (T, 3)

    v, t_min, d = _refine(r, model, (v, t_min, d), jac)
    return float(abs(v)), float(t_min), float(abs(d)), \
        float(np.sqrt(np.mean((r - model((v, t_min, d)))**2)))


def compare_groups(groups):
    """ライン別の d̂ を比べる。groups: {ラインY: [d̂, ...]}

    判定の基準は「群間の差が群内のばらつきを超えているか」。
    n=3 なので検定はせず、差とばらつきの比をそのまま示す。
    """
    keys = sorted(groups)
    stats = {k: (float(np.mean(groups[k])), float(np.std(groups[k], ddof=1))) for k in keys}
    print(f"{'line Y[m]':>10} {'n':>3} {'mean d[m]':>10} {'sd[m]':>8}")
    for k in keys:
        m, sd = stats[k]
        print(f"{k:10.1f} {len(groups[k]):3d} {m:10.3f} {sd:8.3f}")

    if len(keys) == 2:
        (m0, s0), (m1, s1) = stats[keys[0]], stats[keys[1]]
        pooled = np.sqrt((s0**2 + s1**2) / 2)
        gap = abs(m1 - m0)
        print(f"\n群間の差 {gap:.3f} m / 群内ばらつき(pooled) {pooled:.3f} m "
              f"= {gap / max(pooled, 1e-9):.1f} 倍")
    return stats


if __name__ == "__main__":
    # 合成データでの自己検証。実データが来る前に当てはめが正しいことを確かめておく
    rng = np.random.default_rng(0)
    h, y_r, v, x0_true = 15.0, 6.0, 2.5, 13.0
    t = np.arange(0, 26 / v, 0.2)                                # (T,) 0.2s 周期
    print("=== 合成データによる自己検証 ===")
    groups = {}
    for y in (0.5, 3.5):
        d_true = np.hypot(y + y_r, h)
        got = []
        for rep in range(3):
            s = v * t + rng.normal(0, 0.05)                      # 走り出しの微小なずれ
            r_true = np.hypot(s - x0_true, d_true)
            # 観測は 0.846m ビンに量子化され、さらにピーク推定の誤差が乗る
            r_obs = np.round(r_true / 0.846) * 0.846 + rng.normal(0, 0.15, t.size)
            x0_f, d_f, rms = fit_with_s(r_obs, s)
            v_f, tm_f, d_g, rms_g = fit_const_speed(r_obs, t)
            got.append(d_g)
            if rep == 0:
                print(f"  Y={y}  真値 d={d_true:.3f}  X0={x0_true:.1f}  v={v:.2f}")
                print(f"    s既知   : X0={x0_f:6.2f}  d={d_f:6.3f}  rms={rms:.3f}")
                print(f"    s未知   : v={v_f:5.2f}  d={d_g:6.3f}  rms={rms_g:.3f}")
        groups[y] = got
    print()
    compare_groups(groups)
