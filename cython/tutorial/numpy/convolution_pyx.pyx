# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
def naive_convolve(f, g):
    # f は画像で、 (v, w) でインデクスする
    # g はフィルタカーネルで、 (s, t) でインデクスする
    #   ディメンジョンは奇数でなくてはならない
    # h は出力画像で、 (x, y) でインデクスする
    #   クロップしない
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    # smid と tmid は中心ピクセルからエッジまでのピクセル数、
    # つまり 5x5 のフィルタならどちらも 2
    #
    # 出力画像のサイズは入力画像の両端に smid と tmid を足した
    # 値となる
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2*smid
    ymax = wmax + 2*tmid
    # 出力画像の確保
    h = np.zeros([xmax, ymax], dtype=f.dtype)
    # コンボリューションの演算
    for x in range(xmax):
        for y in range(ymax):
            # (x,y) におけるピクセル値 h を計算。
            # フィルタ g の各ピクセル (s, t) に対するコンポーネン
            # トを加算
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h
