#coding: utf-8

from urllib2 import urlopen
import urllib
import os
import re
import htmllib
import httplib

## url name and path
url_name = "www.nikki.ne.jp"
path = "/"


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

## http header
headers = {
    #"Content-Type": "application/x-www-form-urlencoded",
    "Content-Type": "application/x-www-form-urlencoded",
    "Connection": "keep-alive", 
    "Accept": "text/html",
    "Cookie": "JSESSIONID=F02tP4dL1VBZfppfcltv2JnQkW6QjGrhtwpWW9CJTzDh0gnBQMhF!-1729807599; Xlng=Aa8q1nPY239Q-5W4ekO5O5ZNXFlu5LtYJnA9F2O8wz5NF1c48A~~; Ib=AgXajsAqAb-F9zCb-QPF-UMNf-5WBbCoFo5OW42P; Ia=A1Qz4p0fJbUH0qbO9tVXe_csB5cLPopD6Ees7Zp9; shrimp=b; a=201204011818569509; cuid=120401_181856_9867_5a; pitto=0f594c0497601e4f921102f30fc466d4; resque=6b47a0777b9eff379eb67efb9450c9b1; __utma=119409046.291397182.1333271937.1333271937.1333271937.1; __utmb=119409046.2.10.1333271937; __utmc=119409046; __utmz=119409046.1333271937.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=%E6%A5%BD%E5%A4%A9%E3%80%80%E6%B0%91%E8%A1%86; s_pres=%20s_nr%3D1333271978356-New%7C1341047978356%3B; s_sess=%20s_cc%3Dtrue%3B%20scctq%3D1%3B%20s_sq%3D%3B"
}

## connect server of url name
con = httplib.HTTPSConnection(url_name, 443)

## request 1
con.request("GET", "/es/8306/")
#con.request("POST", "?action=login&SERVICE_ID=1", params_post, headers)
res = con.getresponse()
fp = file(os.path.join(os.path.abspath(""), "nikki.txt"), "w" )

for line in list(res.read().split("\n")):
    fp.write(line + "\n")




fp.close()
con.close()

