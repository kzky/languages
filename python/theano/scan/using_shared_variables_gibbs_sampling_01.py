import theano

a = theano.shared(1)
values, updates = theano.scan(lambda: {a: a+1}, n_steps=10)

b = a + 1
c = updates[a] + 1
f = theano.function([], [b, c], updates=updates)

# Not Call
print ("### Not Call ###")
print(b.eval())
print(c.eval())
print(a.get_value())

# Call
print ("### Call ###")
f()
print(b.eval())
print(c.eval())
print(a.get_value())

del a, b, c, f

# Not pass updates
print("### Not pass updates ###")
a = theano.shared(1)
values, updates = theano.scan(lambda: {a: a+1}, n_steps=10)

b = a + 1
c = updates[a] + 1
f = theano.function([], [b, c])

print(b.eval())
print(c.eval())
print(a.get_value())
