# C の関数を呼び出す

C関数を呼び出す基本の話．

- 標準のCライブラリなら*cimport*でよびだせる
- *cimport*可能なファイル一覧は，cythonソースの*Cython/Includes*をみる
- 共有ライブラリは，例えば，libm.soなら，*distutilsのExtention(..., libraries=["m"])*
- 宣言は，*.pxd*に書く(と思っておく)
```cpython
cdef extern from "math.h"
    double sin(double)
    ...
```

# C ライブラリを使う

外部のCライブラリを使う話．自作ライブラリとかサードパーティのライブラリ．

- headerの内容を*c{header_filename}.pxd*に，ほぼコピー
- cでのboolは，cythonで*bint*を使う
- .pxdは，各ライブラリで１つ，または，ヘッダ毎，機能毎に作る
- cython提供の.pxdファイル一覧は，cythonソースの*Cython/Includes*をみる
- *{filename}.pxd*は，*{filename}.pyx*の宣言を書く，cythonは自動的に関連付ける
- headerラップに使った*c{header_filename}.pxd*にはCライブラリの宣言が入っているので，*c{header_filename}.pyx*というファイルは作らない
- *cdef*フィールドの初期値化には*\_\_cinit\_\_*を利用する
- *\_\_cinit\_\_*は，*\_\_init\_\_*よりも先に呼び出される
- *\_\_cinit\_\_*の中では，通常のフィールドは*self*でアクセスしてはならない
- *\_\_cinit\_\_*のシグニチャは，*\_\_init\_\_*と一緒にする
- 必要な場合は，*\_\_dealloc\_\_*でメモリ解放を行う
- C API/pyhon APIの両方を持たせたいなら，*cdef*を *cpdef*にする
- python定義の型と互換性のない型が引数にある場合は，*cpdef*は使えない

# 拡張型の定義 (cdef クラスの定義)
- cdef classで拡張型(cdef classそのもの)の定義ができる
- python classよりもフィールドアクセスが高速
- 任意のCのデータ型をpythonラッパーなしにフィールドに保存可能
- cdef class からpython classの導出は可能，逆はだめ
- mix-inできない
- cdef classのフィールドはコンパイル時に予め定義しなければならない
- 通常はcythonからのみアクセス可能，cdef *public*をつけるとpythonからも可能
- *property* var_nameと書いてもpythonから見える

# pxd ファイル
- headerファイルと思っておけば良い．コードの一部が入っていても良い．
- .pyxから呼ぶには*cimport {pxe_filename}*とする
- *cdef class*のフィールドを宣言する
- 基本的用途は3つ
  - 外部公開用のCの宣言を共有する
  - Cコンパイラにインライン化させたい関数を定義する*cdef inline*と書く．
  - cython用のインターフェイスを作る．モジュール間で共有できるのでpython経由より高速

# プロファイリング
- ファイル全体で有効化したい場合は，*# cython: profile=True*ディレクティブをファイルのトップに書く
- 関数単位では，*@cython.profile(False)*

# unicode と文字列の扱い

現状は，Skip...

# pure Python モード

- pure Pythonモードでは*cython.xxx*でいう特別な関数や*@cython.xxx*という特別なデコレータを使って，*cdef, cpdef*等を行う．

逆に，使いにくくなっている気がしているのでSkip

# NumPy を扱う
- (おそらく)*pip install numpy*でインストールしていると，/usr/includeにヘッダもろもろが存在しないので，*/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/*からコピーする．
- numpyで片付け可能, e.g., *np.ndarray obj*
- numpy objectの中身に対する型付けnp.ndarry[*type*, *ndim*]を行う
- [サンプル](https://github.com/kzky/languages/tree/master/cython/tutorial/numpy)を書いてみたが，sliceとさらなる最適化では早くなったり早くならなかったりする．なぜ?

# Cython に関係のあるソフトウェア
- Pyrex: Cythonの前進
- ctypes: 標準ライブラリであり，.soを簡単にラップ可能．パフォーマンスは悪い
- SWIG: 汎用的なラッパコードジェネレーター．薄いラッパーは作れるが，機能的ラッパーは作れない
- ShedSkin: pythonコードからC++プログラムを生成できるが，ネイティブでサポートされていない操作ができない，および，pythonの標準モジュールをわずかしかサポートしていない

# 参考
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/external.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/clibraries.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/cdef_classes.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/pxd_files.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/profiling_tutorial.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/strings.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/pure.html
- http://omake.accense.com/static/doc-ja/cython/src/tutorial/numpy.html
 -http://omake.accense.com/static/doc-ja/cython/src/tutorial/related_work.html

