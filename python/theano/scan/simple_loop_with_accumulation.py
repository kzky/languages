import theano
import theano.tensor as T

k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

# Optimization saving memory.
final_result = result[-1]

# Compiled function that returns A**k
power = theano.function(inputs=[A, k], outputs=final_result, updates=updates)

print power(range(10), 2)
print power(range(10), 4)
