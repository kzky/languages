# Cythonでできるのこと
- .pyxをcコードにする
- Cythonの変数にcの型をもたせられる
- C/C++ APIの利用

要するに，pythonぽく書けるので生産性が高くて，高速化も可能．

# Installation

```python
pip install cython
```

#  Hello World

[ここ](https://github.com/kzky/languages/tree/master/cython/helloworld)と[ここ](https://github.com/kzky/languages/tree/master/cython/fibonacci)に書いてみた．

# コンパイル方法

2通りある．

- pyximport: 外部Cライブラリや特殊なビルド設定が必要ない場合，cythonモジュールがかなり小さい場合等に使用
- setup.py: 上記以外の場合．自作のCライブラリに依存している等の場合．Makefileと思っておけばいい．

他にも，cythonコマンドを叩いてcコードを生成してから，gccするというのもあると思うが，ここでは扱わない．

## pyximport

```python
import pyximport; pyximport.install(pyimport = True)
import (basename ${filename.pyx} .pyx)
```

な感じで使う．

## setup.py

```python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("helloworld", ["helloworld.pyx"])]
)
```

な感じのコードを書く．

build_extはbuildコマンドで，Extensionは，この名前の.soができる．

使い方は，

```sh
python setup.py build_ext [--inplace]
```

# 静的片付け

```python
def cdef type func(type param)  # 関数引数の場合

cdef type param0, param1, ...  # 変数の場合

```

とかく．

[サンプル](https://github.com/kzky/languages/tree/master/cython/speedup_exmaple)を書いた.


### 結果

```text
kzk@localhost:~/languages/cython/speedup_exmaple
$ python main.py
666666.686668
Python: 0.354106903076 [s]
666666.686668
Cython: 0.196939945221 [s]
666666.686668
cdef Cython: 0.0480411052704 [s]
666666.686668
cdef v2 Cython: 0.00502610206604 [s]

### Ratio Comparison### 
Python: 1.0 [s]
Cython: 1.79804509785 [s]
cdef Cython: 7.37091499213 [s]
cdef v2 Cython: 70.4535837958 [s]
```

すげーはやい．

C/C++ APIの複雑なラップ等は，[チュートリアル，ユーザガイド，リファレンス](http://omake.accense.com/static/doc-ja/cython:)を見ながらやればいい．

現状Cythonを使うというユースケースで思いつくのは，
- pythonのbottle neck code
- GPU Computing


# 参考
- http://omake.accense.com/static/doc-ja/cython/src/userguide/tutorial.html
- http://omake.accense.com/static/doc-ja/cython/
- http://omake.accense.com/static/doc-ja/cython/src/userguide/source_files_and_compilation.html#compilation
