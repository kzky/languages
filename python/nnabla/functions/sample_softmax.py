import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from nnabla.contrib.context import extension_context

def main():
    # Data and Variables
    x_data = np.arange(4*5).reshape((4, 5))
    print(x_data)

    x = nn.Variable(x_data.shape)
    x.d = x_data

    # Context, Graph, Forward
    ctx = extension_context("cpu")
    with nn.context_scope(ctx):
        z = F.softmax(x, axis=1)
    z.forward()

    #  Check
    print("z.shape={}".format(z.shape))
    print("z.d")
    print(z.d)

if __name__ == '__main__':
    main()
