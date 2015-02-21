import pinject


class ClassOne(object):
    def __init__(self, foo):
        self.foo = foo
        pass
    pass

    
class BindingSpecOne(pinject.BindingSpec):
    def configure(self, bind):
        bind('foo', to_instance='foo-')
        pass
    pass


class ClassTwo(object):
    def __init__(self, class_one, bar):
        self.foobar = class_one.foo + bar
        pass
    pass


class BindingSpecTwo(pinject.BindingSpec):
    def configure(self, bind):
        bind('bar', to_instance='-bar')
    pass

    def dependencies(self):
        return [BindingSpecOne()]
    pass

obj_graph = pinject.new_object_graph(binding_specs=[BindingSpecTwo()])
class_two = obj_graph.provide(ClassTwo)
print class_two.foobar
