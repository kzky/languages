import pinject


class SomeClass(object):
    def __init__(self, long_name, long_name2, outer_class):
        self.long_name = long_name
        self.long_name2 = long_name2
        self.outer_class = outer_class
        pass
    pass


class SomeReallyLongClassName(object):
    def __init__(self):
        self.foo = 'foo'
        pass
    pass


class SomeReallyLongClassName2(object):
    def __init__(self):
        self.hoge = 'hoge'
        pass
    pass


class InnerClass(object):
    """
    """
    
    def __init__(self, ):
        """
        """
        self.name = "inner_class"
        pass
    pass


class OuterClass(object):
    """
    """
    
    def __init__(self, inner_class):
        """
        """
        self.inner_class = inner_class
        pass
    pass


class MyBindingSpec(pinject.BindingSpec):
    def configure(self, bind):
        bind('long_name', to_class=SomeReallyLongClassName)
        bind('long_name2', to_class=SomeReallyLongClassName2, )
        bind('outer_class', to_class=OuterClass)
        #bind('inner_class', to_class=InnerClass)
        pass
    pass
    
obj_graph = pinject.new_object_graph(binding_specs=[MyBindingSpec()])
some_class = obj_graph.provide(SomeClass)
print some_class.long_name.foo
print some_class.long_name2.hoge
print some_class.outer_class.inner_class.name

