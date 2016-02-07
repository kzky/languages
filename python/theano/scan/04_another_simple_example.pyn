import numpy as np
import theano
import theano.tensor as T

location = T.imatrix("location")
values = T.vector("values")
output_model = T.matrix("output_model")

def set_value_at_position(a_location, a_value, output_model):
    zeros = T.zeros_like(output_model)
    zeros_subtensor = zeros[a_location[0], a_location[1]]
    return T.set_subtensor(zeros_subtensor, a_value)

result, updates = theano.scan(fn=set_value_at_position,
                              outputs_info=None,
                              sequences=[location, values],
                              non_sequences=output_model)

assign_values_at_positions = theano.function(inputs=[location, values, output_model], outputs=result)

# test
test_locations = np.asarray([[1, 1], [2, 3]], dtype=np.int32)
test_values = np.asarray([42, 50], dtype=np.float32)
test_output_model = np.zeros((5, 5), dtype=np.float32)
print assign_values_at_positions(test_locations, test_values, test_output_model)
