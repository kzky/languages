#!/usr/bin/python

## solrpyを使ったほうが早いかも
## solrは転置インデックスで検索

import urllib2
import urllib

## params
count =0
xml_add_tmp = """<add>
%s</add>"""
xml_doc_tmp = """<doc>
<field name='id'>%s</field>
<field name='title'>%s</field>
<field name='content'>%s</field>
<field name='date'>%s</field>
</doc>
"""
url = "http://localhost:8983/solr/drbd/update"
headers = {"Content-type": "application/xml"}
xml_docs_tmp = ""
fpin = None 

## read and post
print "START"
try:
    fpin = open("/home/kzk/dataset/drbd/73789/contents.tsv", "r")
    for line in fpin:
        print count
        
        ## skip 
        if count < 2:
            count+=1
            continue
        
        ## END
        if line == "END":
            break;
        
        attr = line[:-1].split("\t")
        ## create date
        if attr[16] != "":
            date = attr[16].replace("/", "-") + "T00:00:00Z"
        else:
            date = "2000-09-03T00:00:00Z"

        ## create params
        title = attr[2].replace("<br>", "").replace("&", "").replace("/", "").replace(">", "").replace("<", "")
        content = attr[10].replace("<br>", "").replace("&", "").replace("/", "").replace(">", "").replace("<", "")
        xml_docs_tmp += xml_doc_tmp % (attr[0], title, content, date)

        ## post
        if count%100 == 0:
            ## create params
            data = xml_add_tmp % xml_docs_tmp
            #print data
            req = urllib2.Request(url=url, data = data, headers=headers)
            res = urllib2.urlopen(req);
            xml_docs_tmp = ""


        count+=1
    
except IOError:
    print "no iput file"
    exit(1);
finally:
    data = xml_add_tmp % xml_docs_tmp
    req = urllib2.Request(url=url, data=data, headers=headers)
    urllib2.urlopen(req);
    fpin.close()

urllib2.urlopen(url+"?softCommit=true")
print "END"
