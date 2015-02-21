class Singleton(object):
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance == None:
            cls.instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.instance

if __name__ == '__main__':
    
    s1 = Singleton()
    s2 = Singleton()
    print id(s1) == id(s2)
