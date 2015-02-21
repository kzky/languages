import pinject


class Foo(object):
    def __init__(self):
        self.forty_two = 42
        pass
    pass


class SomeBindingSpec(pinject.BindingSpec):
    def configure(self, bind):
        bind('foo', to_class=Foo, in_scope=pinject.PROTOTYPE)
        pass
    pass


class NeedsProvider(object):
    def __init__(self, provide_foo):
        self.provide_foo = provide_foo
        pass

    #def __init__(self, foo):
    #    self.foo = foo
    #    pass

    pass


obj_graph = pinject.new_object_graph(binding_specs=[SomeBindingSpec()])
needs_provider = obj_graph.provide(NeedsProvider)
print needs_provider.provide_foo() is needs_provider.provide_foo()
#print needs_foo is needs_provider.foo
