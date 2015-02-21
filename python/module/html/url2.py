#coding: utf-8

import cookielib
import urllib, urllib2, os

#url_name = "https://www.nikki.ne.jp/?action=login&SERVICE_ID=1"
#params_post = urllib.urlencode({
#        "__event": "ID01_001_001",
#        "service_id": "p06",
#        "return_url": "index.phtml",
#        "return_url_nikki": "http://www.nikki.ne.jp/a/login/", 
#        "eVar22": "www:top_limit:rakutenLogin_btn",
#        "u": "rkzfilter@gmail.com", 
#        "p": "harumaki51",
#        "submit": "ログイン"
#        })
#
#cj = cookielib.CookieJar()
#opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
#
#url = opener.open(url_name, params_post)
#data=url.read()
#print url.geturl()
#
#fp = file(os.path.join(os.path.abspath(""), "url2.txt"), "w")
#print >> fp, data
#

url_name = "http://grp03.id.rakuten.co.jp/rms/nid/login"
#url_name = "https://www.nikki.ne.jp/?action=login&SERVICE_ID=1"
params_post = urllib.urlencode({
        "__event": "ID01_001_001",
        "service_id": "p06",
        "return_url": "index.phtml",
        "return_url_nikki": "http://www.nikki.ne.jp/a/login/", 
        "eVar22": "www:top_limit:rakutenLogin_btn",
        "u": "rkzfilter@gmail.com", 
        "p": "harumaki51",
        "submit": "ログイン"
        })

cj = cookielib.CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))

url = opener.open(url_name, params_post)
data=url.read()
print url.geturl()

fp = file(os.path.join(os.path.abspath(""), "url2.txt"), "w")
print >> fp, data

