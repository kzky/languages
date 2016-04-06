#!/usr/bin/env python

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def main():
    D = get_class("datetime.datetime")
    print D
    print D.now()
    pass


if __name__ == '__main__':
    main()
