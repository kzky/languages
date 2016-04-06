#!/usr/bin/env python
"""
Join Operation for embedded document field
"""

from mongoengine import *

connect("sample01")

class A(EmbeddedDocument):
    """
    """
    a = StringField()
    
class B(EmbeddedDocument):
    """
    """
    b = StringField()
    a = EmbeddedDocumentField(A)

class C(Document):
    """
    """
    c = StringField()
    b = EmbeddedDocumentField(B)

def main():

    C.drop_collection()
    
    a = A(a="a")
    b = B(b="b", a=a)
    c = C(c="c", b=b)
    c.save()

    c_ = C.objects(b__a__a="a").first()
    print c_.b.a.a
    
    pass

if __name__ == '__main__':
    main()
