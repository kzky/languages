import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from nnabla.contrib.context import extension_context

def main():
    # Data and Variables
    x_data = np.random.rand(4, 5)
    y_data = np.random.rand(5, 4)

    x = nn.Variable(x_data.shape)
    y = nn.Variable(y_data.shape)
    x.d = x_data
    y.d = y_data

    # Context, Graph, Forward
    ctx = extension_context("cpu")
    with nn.context_scope(ctx):
        z = F.affine(x, y)
    z.forward()

    #  Check
    print("z.shape={}".format(z.shape))
    print("z.d")
    print(z.d)
    print("numpy.dot")
    print(x_data.dot(y_data))


if __name__ == '__main__':
    main()
