#coding: utf-8

import urllib


params = urllib.urlencode({
        "hl": "ja",
        "gl": "tbm",
        "nws": "nws",
        "q": "sony"
})
urlname = "https://www.google.co.jp/search?%s" % params

# google検索はAPIを使わないといけないらしい
# searching with scrapers is forbidden

print urlname

url = urllib.urlopen(urlname)

print url.read()

print ".........."


