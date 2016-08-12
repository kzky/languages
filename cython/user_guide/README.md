## 概要
- pythonぽく書けて高生産性
- 型付けすることによる高速化
- C/C++の中間言語的な感じ
- cythonコードから作成されるCコードをCで使える
- C/C++のコードをcythonで使える．
- すなわち，pythonで利用可能

## チュートリアル基礎編
- [ここ](https://github.com/kzky/languages/blob/master/cython/README.md)の内容と一緒．

##  基本の言語仕様
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

##  拡張型
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

## 上記を使う関数
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

##  拡張型の特殊メソッド
- *\_\_cinit\_\_*は必ず1度だけ呼び出される
- オブジェクトアトリビュートはメモリ領域が確保されて，Cの型のアトリビュートは0かNULLで初期値化されている．pythonオブジェクトはNoneで初期値化されている
- 継承がある場合，*\_\_cinit\_\_*は親から呼ばれ，親クラスの*\_\_cinit\_\_*は小クラスで明示的に呼べない
- コンストラクタに渡した引数は*\_\_cinit\_\_*と *\_\_init\_\_*両方に渡される
- 設計として*\_\_cinit\_\_*に *\*args, \*\*kwargs*を用意しておいたほうがいい
- *\_\_dealloc\_\_*で *\_\_cinit\_\_*
- 細かい話は[doc](http://omake.accense.com/static/doc-ja/cython/src/userguide/special_methods.html)参照

##  Cython モジュール間で宣言を共有する
- *.pxd*のこと
- Cの型定義，externのC関数や変数の宣言，モジュールで定義したC関数の宣言，拡張型の定義部分が入る
- 他のモジュールから定義ファイルを使いたい場合は定義ファイルの*.pxd*を作る
- *.pxd*ファイルのimportは*cimport module_name*と書く
- コンパイル時には*-I*オプションを使ってファイルを指定する

##  外部の C のコードにインタフェースする
- externでCコードを利用できる
- publicでcythonコードをCで利用できる
- publicを使うことはあまりないかも
- externでは，*cdef extern from "header_name.h"*と書いてCのヘッダの内容をほぼコピペすればいい．詳細は[ドキュメント](http://omake.accense.com/static/doc-ja/cython/src/userguide/external_C_code.html)を見ながらやったほうがいい

##  ソースファイルとコンパイル
- [ここ](https://github.com/kzky/languages/tree/master/cython)にまとめたのとほぼ同じ

##  アーリーバインディングによる高速化
- 高速化したいならちゃんと型付きで宣言，アーリーバインディングしましょうという
- 高速化のために，関数/メソッドなら，パラメータおよび戻り値にに型をつける
-  高速化のために，ローカル変数にも型をつける．特にloopの時
- numpyのndarrayでも型をつけられる．さらに，numpy ndarray objectの中身に対しても型付けが可能

##  Cython から C++ を使う
- v0.13からさらに[機能](http://omake.accense.com/static/doc-ja/cython/src/userguide/wrapping_CPlusPlus.html#c-cython-v0-13)が追加された
- C++のラップ手順は，Cのラップにかなり似ている
  - setup.pyスクリプト全体か，ソースファイル指定オプションで言語をC++に指定する
  - *.pxd*フィアルを作成し, *cdef extern from {header\_name}.h namespace {name}*を指定
  - *cdef cppclass:*ブロックでクラスの宣言
  - *public*にする変数，コンストラクタ，メソッドを宣言
  - 拡張モジュールを書いて*.pxd*を *cimport*する
- setup.pyの書き方は基本以下のとおり, cythonize関数を使う
```python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(
        "rect.pyx",                                 # cython module
        sources=["Rectangle.cpp"],  # c code
        language="c++",                      # lang-specified
))
```
- この場合は，*rect.cpp*というソースファイルを生成してからコンパイルし，次に*Rectanble.cpp*をコンパイルしてから，2つのオブジェクトファイルを*rect.so*にリンクする
- 他に勘弁な記述も可能
```python
from distutils.core import setup
from Cython.Build.import cythonize

setup(
    name="rectangleapp"
    ext_module= cythonize("*.pyx")
```
- ただしこの場合は，*.pyx*ファイルにコメントブロックを書かなければならない
```
# distutils language = c++
# distutils sources = Rectangle.cpp
```
- ラップするには，まずC++のクラスインターフェース宣言をする
```cython
# rect.pxd
cdef extern from "Rectangle.h" namespace "shapes":
  cdef cppclass Rectangle:
    Rectanble(int, int, int, int) except +
    int x0, y0, x1, y1
    int getLength()
    int getHeight()
    int getArea()
    void move(int, int)
```
- コンストラクタででた例外を伝播させるために*except +*って書くのがポイント
- 次にCythonでラッパークラスをつくる．*cdef class*でラップする. 
- *\_\_cinit\_\_*でC++のクラスをつくり，*\_\_dealloc\_\_*でdeleteするのが鉄板のやり方．アトリビュートに直接アクセスしたかったら*property var:*ブロックをつくる．
```
# rect.pyx
#from rect cimport Rectangle  # なくてもいい

cdef class PyRectangle:
    cdef Rectangle *thisptr # ラップ対象の C++ インスタンスを保持する
    def __cinit__(self, int x0, int y0, int x1, int y1):
        self.thisptr = new Rectangle(x0, y0, x1, y1)
    def __dealloc__(self):
        del self.thisptr
    def getLength(self):
        return self.thisptr.getLength()
    def getHeight(self):
        return self.thisptr.getHeight()
    def getArea(self):
        return self.thisptr.getArea()
    def move(self, dx, dy):
        self.thisptr.move(dx, dy)

    property x0:
        def __get__(self): return self.thisptr.x0
        def __set__(self, x0): self.thisptr.x0 = x0
```
- もっと細かい話(オーバーロード，演算子のオーバーロード，入れ子クラス，テンプレート，例外翻訳表，静的メンバメソッド)は[ドキュメント](http://omake.accense.com/static/doc-ja/cython/src/userguide/wrapping_CPlusPlus.html#id3)参照．


##  融合型 (テンプレート)
- v0.25 (20160810時点で最新)でも実験的なのでbugがあるかも
- なので，とりあえずskip

##  Cython コードを PyPy に移植する
- PyPyを使う予定は今のところないので，とりあえずskip

##  Cython の制約
- とりあえずskip

##  Cython と Pyrex の違い
- とりあえずskip

##  型付きメモリビュー (Typed Memoryview)
- numpy arrayのようなバッファに高速にpython overheadを伴わずにアクセス可能
- numpyのアレイバッファーサポート(np.ndarray[np.int_t, ndim=2])に似ているが，より機能が豊富でクリーンなコードが書ける
- C Array, Cython Array, Numpy Arrayがあって，実体を相互にそれぞれのViewに入れられる．
- 詳細は[ドキュメント](http://omake.accense.com/static/doc-ja/cython/src/userguide/memoryviews.html#view-cython-arrays)参考．

##  並列化

##  Cython プログラムのデバッグ
- GILの開放が可能


## Reference
- http://omake.accense.com/static/doc-ja/cython/index.html
