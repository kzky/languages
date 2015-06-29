import numpy as np
import theano
import theano.tensor as T
import time
import csv


def compute_logistic_regression(Ns=None, feats=None, output_filepath="/tmp/logreg.out"):
    """
    """

    xs = []
    if type(Ns) == list and type(feats) == list:
        return
    elif type(Ns) == list:
        xs = Ns
        #feats = [784]  # default used in the original example
        feats = [64000]
    elif type(feats) == list:
        xs = feats
        Ns = [400]  # default used in the original example
    else:
        return

    with open(output_filepath, "w") as fpout:
        writer = csv.writer(fpout, delimiter=",")
        cnt = 0
        for N in Ns:
            for feat in feats:
                rng = np.random
                D = (rng.randn(N, feat).astype(theano.config.floatX),
                     rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
                training_steps = 100  # changed from the original example
                 
                # Declare Theano symbolic variables
                x = T.matrix("x")
                y = T.vector("y")
                w = theano.shared(rng.randn(feat).astype(theano.config.floatX), name="w")
                b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")
                x.tag.test_value = D[0]
                y.tag.test_value = D[1]
                #print "Initial model:"
                #print w.get_value(), b.get_value()
                 
                # Construct Theano expression graph
                p_1 = 1 / (1 + T.exp(-T.dot(x, w)-b)) # Probability of having a one
                prediction = p_1 > 0.5 # The prediction that is done: 0 or 1
                xent = -y*T.log(p_1) - (1-y)*T.log(1-p_1) # Cross-entropy
                cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
                gw,gb = T.grad(cost, [w,b])
                 
                # Compile expressions to functions
                train = theano.function(
                    inputs=[x, y],
                    outputs=[prediction, xent],
                    updates={w: w-0.01*gw, b: b-0.01*gb},
                    name = "train")
                predict = theano.function(inputs=[x], outputs=prediction,
                                          name = "predict")
                 
                if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
                        train.maker.fgraph.toposort()]):
                    print 'Used the cpu'
                elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
                        train.maker.fgraph.toposort()]):
                    print 'Used the gpu'
                else:
                    print 'ERROR, not able to tell if theano used the cpu or the gpu'
                    print train.maker.fgraph.toposort()

                print "count = ", cnt
                st = time.time()
                batch_samples = 5000
                for i in range(training_steps):
                    for k in np.arange(0, D[0].shape[0], batch_samples):
                        print k, k+batch_samples
                        pred, err = train(D[0][k:(k+batch_samples), :], D[1][k:(k+batch_samples)])
                        pass
                        
                #print "Final model:"
                #print w.get_value(), b.get_value()
                 
                #print "target values for D"
                #print D[1]
                # 
                #print "prediction on D"
                #print predict(D[0])
                
                et = time.time()
                dt = et - st

                # write
                writer.writerow([xs[cnt], dt])
                cnt += 1
                pass
            pass

def main():

    output_filepath = "/tmp/logreg_Ns2.out"
    Ns = list(np.arange(10000, 70000, 10000))
    compute_logistic_regression(Ns=Ns, output_filepath=output_filepath)

    #output_filepath = "/tmp/logreg_feats.out"
    #feats = list(np.arange(10000, 70000, 10000))
    #compute_logistic_regression(feats=feats, output_filepath=output_filepath)

    pass

if __name__ == '__main__':
    main()
