import pinject


class ClassWithTediousInitializer(object):
    @pinject.copy_args_to_internal_fields
    def __init__(self, foo, bar, baz, quux):
        pass

cwti = ClassWithTediousInitializer('a-bar', 'a-foo', 'a-baz', 'a-quux')
print cwti._foo



