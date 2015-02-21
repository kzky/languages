#coding: utf-8

import urllib2
from htmllib import HTMLParser
from formatter import NullFormatter

url_name = "http://www.python-izm.com/"
html_data = urllib2.urlopen(url_name)

## HTMLParser クラスを上書きする
## 解析したいように関数を再定義する

## lxml, libxml2というxpath langを利用したモジュールがある, 前者のが楽？

class TestParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self, NullFormatter())
        self.link = []
        
    def handle_starttag(self, tagname, method, attribute):
        if tagname.lower() == "a":
            for i in attribute:
                if i[0].lower() == "href":
                     self.link.append(i[1])

parser = TestParser()
url = parser.feed(html_data.read())


parser.close();
html_data.close()


