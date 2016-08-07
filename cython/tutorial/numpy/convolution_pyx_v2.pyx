# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
# "cimport" は、 numpy モジュールを使うコードのコンパイル時に必要
# な情報を import するのに使います。
# (この情報は numpy.pxd に入っています。現状、 numpy.pxd は Cython
# ディストリビューションに入っています)
cimport numpy as np
# 変数 DTYPE を使って、アレイのデータ型に手を加えます。DTYPE には、
# NumPy のランタイム型情報オブジェクト (runtime type info object)
# を代入します。
DTYPE = np.int
# "ctypedef" は、コンパイル時に決定した DTYPE_t で参照できるよ
# うにします。numpy モジュールのデータ型には、すべて _t という
# サフィックスのついたコンパイル時用の型があります。
ctypedef np.int_t DTYPE_t
# "def" は引数の型指定を行えますが、戻り値の型を指定できません。
# "def" 関数型の引数の型は、関数に入るときに動的にチェックされます。
#
# アレイ f, g, h は "np.ndarray" インスタンス型に型付けされていま
# す。その効果は a) 関数の引数が本当に NumPy アレイかどうかチェッ
# クが入ることと、 b) f.shape[0] のようなアトリビュートアクセスの
# 一部が格段に効率的になること、だけです (この例の中では、どちらも
# さして意味はありません)。
def naive_convolve(np.ndarray f, np.ndarray g):
    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")
    assert f.dtype == DTYPE and g.dtype == DTYPE
    # "cdef" キーワードは、関数の中で変数の型を宣言するのにも使い
    # ます。 "cdef" は、関数内のトップインデントレベルでしか使えま
    # せん (他の場所で使えるようにするのは大したことではありません。
    # もしいい考えがあったら提案してください)。
    #
    # インデクスには、 "int" 型を使います。この型は C の int 型に
    # 対応しています。 ("unsigned int" のような) 他の型も使えます。
    # アレイのインデクスとして適切な型を使いたい純粋主義の人は、
    # "Py_ssize_t" を使ってもかまいません。
    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2*smid
    cdef int ymax = wmax + 2*tmid
    cdef np.ndarray h = np.zeros([xmax, ymax], dtype=DTYPE)
    cdef int x, y, s, t, v, w
    # 変数全てについて型を定義するのがとても大事です。型を定義し忘
    # れても何の警告もでませんが、(変数が暗黙のうちに Python オブ
    # ジェクトに片付けされるので) コードは極端に遅くなります。
    cdef int s_from, s_to, t_from, t_to
    # 変数 value に対しては、アレイに保存されいているのと同じデー
    # タ型を使いたいので、上で定義した "DTYPE_t" を使います。
    # 注! この操作には、重大な副作用があります。 "value" がデータ
    # 型の定義域をオーバフローすると、 Python のように例外が送出さ
    # れるのではなく、 C のときと同様に単なる桁落ち (wrap around)
    # を起こします。
    cdef DTYPE_t value
    for x in range(xmax):
        for y in range(ymax):
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
