# 概要
- pythonぽく書けて高生産性
- 型付けすることによる高速化
- C/C++の中間言語的な感じ
- cythonコードから作成されるCコードをCで使える
- C/C++のコードをcythonで使える．
- すなわち，pythonで利用可能

# チュートリアル基礎編
- [ここ](https://github.com/kzky/languages/blob/master/cython/README.md)の内容と一緒．

#  基本の言語仕様
- cdefは，Cの変数を宣言するのに使用する
- e.g., *cdef int *name, *cdef struct* name, *cdef union* name, *cdef enum* name
- *ctypedef* type ailasも使える
- *cdef:*でブロック的な感じで複数変数を宣言可能
```cython
cdef:
    struct Spam:
        int tons
    int i
    float f
    Spam *p
    void f(Spam *s):
        print s.tson, "Tons of spam"
```
- *cdef*で宣言した関数は，cの値のパラメータ/python objを引数と戻り値に取れる
- cython moduleの中ではpython/cの関数は相互に自由に呼び出せる
- pythonからcython moduleで定義した関数を呼び出すには，*def*で定義しなくてはならない．なので，*cdef*関数をpythonから呼びたい場合は，*def*でラッパー関数を作るか*cpdef*で関数を作る
- python関数のパラメータをCのデータ型で宣言できて，そのパラメータはpythonオブジェクトとして引き渡され，自動的にCの値に変換される．
- 自動変換は，数値型，文字列型，構造体(中身は数値型，文字列型，および構造体)に限られる[参考](http://omake.accense.com/static/doc-ja/cython/src/userguide/language_basics.html#id3)
- *cdef*で宣言した関数の引数/戻り値が型なしの場合は，python objectとして認識される
- なので，明示的にobjectをつけたほうがよい
```cython
cdef object func(object obj):
    ...
```
- pythonオブジェクトを返さないCの関数がエラーを伝播する場合は関数にexceptをつける
```cython
cdef int fucn(...) except -1:
    ...
```
- エラー値が正常関数の値域全体に含まれている場合は，*cdef func(...) except? -1:*のよう書く
- *cdef func(...) except \**のような書き方も可能だが，この場合はcythonは常に*PyErr_Occurred()*を呼び出す
- C++なら*cdef int func(...) except +*と書く
- *cdef*の関数の例外は，*int, enum, float, pointer type*を返す関数でのみ宣言可能
- *void*関数での例外は*cpdef void func(...) except \**としか書けない
- 文と式は，基本的にpythonと同じ
- キャストは*<type>val*

#  拡張型
- *extension type class*と呼ばれ，cythonのクラスのこと
- *cdef*を使ってアトリビュートを定義できる
- 拡張型のアトリビュートはコンパイル時に決まっていないとならないので，拡張型では動的に追加できない
- ただし，拡張型のpython子クラスを定義すれば，動的にアトリビュートを追加可能
- 拡張型のアトリビュートにpythonコードから直接アクセスするには*public, readonly*をつける
```cython
cdef class Shrubbery:
    cdef public int width, height
    cdef readonly float depth
```
- 拡張型のcdefアトリビュートをpython objにしたいときは*cdef object obj*とする
- 拡張型を関数の引数にする場合には必ず型をつける．さもないとpythonオブジェクトとして認識されて実行時にアトリビュートエラーが起きる
```cython
cdef widen_shrubbery(Shrubbery sh, extra\_width):
    sh.width = sh.width + extra_width
```
- ローカル変数でも同じ
```cythn
cdef Shrubbery another\_shrubbery(Shruberry sh1):
    cdef Shrubbery sh2
    sh2 = Shrubbery()
    sh2.width = sh1.width
    sh2.height = sh2.height
    return sh2
```
- 拡張型を返す関数から直接アトリビュートにアクセスしたい場合はキャストする
```cython
print (<Shrubbery>quest()).width   # 存在しないアトリビュートにアクセスする可能性あり
print (<Shrubbery?>quest()).width  # 型チェックしてからキャスト
```
- 拡張型の特殊メソッドはpythonのと基本同じだが違いがある．[違い](http://omake.accense.com/static/doc-ja/cython/src/userguide/special_methods.html#special-methods)．
- プロパティの実装は*property var*と書く
```cython
cdef class Spam:
    property cheese:
        def __get__(self):
            ...
        def __set__(self, value):
            ...
        def __del(self):
            ...
```
- 拡張型は継承できるが，ミックスインはできない
- *@cython.final*デコレータで拡張できないようになる
- 拡張型メソッドのオーバーライドも可能
- 相互参照している場合は前方宣言すること
- extern拡張型を使うとpythonオブジェクトの内部構造にアクセスしたりcythonを使わない拡張モジュールから拡張型にアクセスしたりできる
- extern拡張型は*cdef extern from "header_name.h"*で囲まれたブロックで宣言する拡張型，または*cdef extern class class_name*ことだと思われる
- ビルトイン複素数オブジェクトのCレベルのメンバを取り出す例
```cython
cdef extern from "complexobject.h":
    struct Py_complex:
        double real
        double imag

    ctypedef class __builtin__.complex [object PyComplexObject]:
        cdef Py_complex cval

# 上記を使う関数
def spam(complex c):
    print "Real: ", c.cval.real
    print "Imag", c.cval.imag
```
- ctypedefしているのは，python headerファイルでPyComplexObject構造体を宣言しているため
- *\_\_builtin\_\_*は，pythonの型オブジェクトが入っているモジュール
- extern拡張型の宣言ではメソッドの宣言はいらない
- クラス宣言の一部*[]*で囲むのはextern/public拡張型でのみ使用可能
- *[object object\_struct\_name, type type\_object\_name]*の前者はC構造体に割り当てられる名前，後者はtypeオブジェクトに割り当てられる名前
- extern拡張クラスの宣言は，モジュール名も含めないとならない．*cdef extern class module_name.class_name:*
- この場合はモジュールロード時に，暗黙で*from module_name import class_name*としている
- public拡張型*cdef public class class_name*を使うと.hファイルができて外部のCコードからcython定義の拡張型が使える．

#  拡張型の特殊メソッド

#  Cython モジュール間で宣言を共有する

#  外部の C のコードにインタフェースする

#  ソースファイルとコンパイル

#  アーリーバインディングによる高速化

#  Cython から C++ を使う

#  融合型 (テンプレート)

#  Cython コードを PyPy に移植する

#  Cython の制約

#  Cython と Pyrex の違い

#  型付きメモリビュー (Typed Memoryview)

#  並列化

#  Cython プログラムのデバッグ
- GILの開放が可能


# Reference
- http://omake.accense.com/static/doc-ja/cython/index.html
