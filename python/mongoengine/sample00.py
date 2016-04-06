#!/usr/bin/env python
"""
Join Operation for reference field
"""

from mongoengine import *

connect("sample00")

class A(Document):
    """
    """
    a = StringField()
    
class B(Document):
    """
    """
    b = StringField()
    a = ReferenceField(A)

class C(Document):
    """
    """
    c = StringField()
    b = ReferenceField(B)


def main():

    A.drop_collection()
    B.drop_collection()
    C.drop_collection()
    
    a = A(a="a")
    a.save()

    b = B(b="b", a=a)
    b.save()

    c = C(c="c", b=b)
    c.save()

    b_ = B.objects(a__a="a").first()
    print b_.b
    print b_.a.a
    
    pass

if __name__ == '__main__':
    main()
