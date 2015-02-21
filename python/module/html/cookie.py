#coding: utf-8

from urllib2 import urlopen
import urllib
import os
import re
import htmllib
import httplib

## url name and path
#url_name = "www.nikki.ne.jp"           ## これだと元のページにredirectされる?
#url_name = "grp03.id.rakuten.co.jp"   ## ここもおかしい？(多分これが認証サーバ)
url_name = "203.190.61.229"

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
#con.request("POST", "/?action=login&SERVICE_ID=1")
#con.request("POST", "/rms/nid/vc", params_post, headers) 

## verisigninに任せている, 認証サーバは楽天のドメイン内にある


con.request("POST", "/rms/nid/login", params_post, headers)
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

fp.close()

## request 2
t = re.search("name=\"t\" value=(?=\"(.*?)\")", data).group(1)
print t ## HTTPSなので、ここは暗号化されている？ そのまま送ってはだめ？


#https://grp03.id.rakuten.co.jp/rms/nid/personalfwd?profile_id=326499073&labelId=1&service_id=p06


params_post = urllib.urlencode({
        "t": t,
        "service_id": "p06",
        "return_url": "index.phtml",
        "return_url_nikki": "http://www.nikki.ne.jp/a/login/", 
        "eVar22": "www:top_limit:rakutenLogin_btn",
        "birthday": "19870412", 
        "submit": "ログイン", 
        "autologout": "true"
        })
cookie =  res.getheader("set-cookie")
xpow = res.getheader("X-Powered-By")
tc = res.getheader("Transfer-Encoding")
p3p = res.getheader("P3P")

## クッキーの設定がよろしくないから、secondarthbirthでエラー？
#print cookie
jid = re.search("(JSESSIONID=.*?);", cookie).group(1)
xlng = re.search("(Xlng=.*?);", cookie).group(1)
headers["Cookie"] = jid + "; " + xlng
#headers["Cookie"] = cookie
#headers["X-Powered-By"] = xpow
#headers["Transfer-Encoding"] = tc
#headers["P3P"] = p3p
#headers["Expires"] = res.getheader("Expires")
#headers["Date"] = res.getheader("Date")
#headers["P3P"] = p3p

con.request("POST", "/rms/nid/secondauthbirth", params_post, headers) ## pathがおかしい？ formのaction と cookieのpathをみると分かる
res = con.getresponse()

print headers
print
print "Second Request"
print res.status, res.reason
print "headers:", "\n" , res.getheaders()
#print res.msg()

data = res.read()

fp = file(os.path.join(os.path.abspath(""), "response2.html"), "w" )
for line in list(data.split("\n")):
    fp.write(line + "\n")

## get

con2 = httplib.HTTPSConnection("www.nikki.ne.jp", 443)
con2.request("GET", "/es/8306/", headers=headers)
res = con2.getresponse()
data = res.read()

fp = file(os.path.join(os.path.abspath(""), "response3.html"), "w" )
for line in list(data.split("\n")):
    fp.write(line + "\n")

fp.close()
con2.close()
con.close()

print t
