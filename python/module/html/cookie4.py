#coding: utf-8

from urllib2 import urlopen
import urllib
import os
import re
import htmllib
import httplib

## url name and path
url_name = "www.nikki.ne.jp"           ## これだと元のページにredirectされる?
#url_name = "grp03.id.rakuten.co.jp"   ## ここもおかしい？(多分これが認証サーバ)


## post data
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

print "params_post:", params_post, "\n"

## http header
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Connection": "keep-alive", 
    "Accept": "text/plain"
}

## connect server of url name
con = httplib.HTTPSConnection(url_name, 443)

## request 1
con.request("POST", "/a/login", params_post, headers)
#con.request("POST", "/rms/nid/login", params_post, headers) ## verisigninに任せている
#con.request("POST", "/rms/nid/login", params_post, headers)
#con.request("POST", "/rms/nid/", params_post, headers)

res = con.getresponse()

print "First Requset"
print res.status, res.reason, "\n"
print "headers:", "\n" , res.getheaders(), "\n"
print "message:", "\n", res.msg, "\n"
print headers, "\n"
data = res.read()

fp = file(os.path.join(os.path.abspath(""), "response1.html"), "w" )
for line in list(data.split("\n")):
    fp.write(line + "\n")

cookie =  res.getheader("set-cookie")
fp.close()
con.close()

