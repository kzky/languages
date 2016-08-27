# Playing with nogil of Cython 

Here I try to speed up some programs in Python that computation becomes overhead using Cython. 

There are other approaches to speed up in Python with either multiprocessing or Numba (JIT compiler). In using multiprocessing for speedup purpose, we have to either use a shared memory which interfaces are a bit hard to use or copy the same data in processes, which uses much more memory than using shared memory. In case of numba, it is some times difficult to make Numba in your environement. In my exerience, when I tried to use numba in ubuntu14.04 and ubuntu16.04, I could not intall via pip; instead, I had to install it by *git clone ...*, and there are many dependencies that it depends on so that just doing *python setup.py* was not enough to completely install.

Cython is one of tools to speed up Python codes except for using either multiprocessing or numba. I tried to speed up some programs which computation is matter step-by-step. Thease codes in this directory are based on those in references. 

## Environemnt

- OS: ubuntu14.04
- CPU: Intel(R) Core(TM) i5-2540M CPU @ 2.60GHz x 2 cores
- Numpy: 1.11.1

## Experiment Settup

Experiments are very simple by following [Parallel computing in Cython - threads](http://nealhughes.net/parallelcomp2/). Task is to compute *f = exp(x)* if *x > 0*, otherwise *0* with *x* being sampled from *2 x N(0, 1) - 1*. Here *x* is a long 1d-array, *7x7x512x4096*, and the task is ran 10 times and report the average and std of the elapsed times. Baseline is computation using numpy.

## Exp. 0

Just use *type* in Cython.

### Resutls

```
# Numpy
Elapsed time (ave) 2.24134101868 [s]
Elapsed time (std) 0.463485348686 [s]
# Cython
Elapsed time (ave) 24.3929426193 [s]
Elapsed time (std) 2.95880068941 [s]
```

Numpy beats the unoptimized Cython code.

### Codes
- [multi_thread00.pyx](./multi_thread00.pyx)
- [main_multi_thread00.py](./main_multi_thread00.py)

## Exp. 1

Optimize Cython codes. 
- Specify *type* and dimentsion of np.ndarray in the arguemnt of the function
- Uncheck the bound


### Resutls

```
# Numpy
Elapsed time (ave) 2.85961380005 [s]
Elapsed time (std) 0.492088334255 [s]
# Cython
Elapsed time (ave) 2.75198538303 [s]
Elapsed time (std) 0.205762149601 [s]
```

In both stats, cython is a bit faster and more stable.

### Codes
- [multi_thread01.pyx](./multi_thread01.pyx)
- [main_multi_thread01.py](./main_multi_thread01.py)

## Exp. 2

Parallelize computation, so use use
- *prange*
- *nogil*

### Resutls

```
$ python main_multi_thread02.py
# Numpy
Elapsed time (ave) 2.58183617592 [s]
Elapsed time (std) 0.508897742858 [s]
# Cython
Elapsed time (ave) 2.57006659508 [s]
Elapsed time (std) 0.376996569113 [s]
```

In both stats, cython is a bit faster and more stable.

### Codes
- [multi_thread02.pyx](./multi_thread02.pyx)
- [main_multi_thread02.py](./main_multi_thread02.py)

## Exp. 3

OpenMP is not good choice to speedup, so use threading with *nogil*

### Resutls

```
$ python main_multi_thread03.py
# Numpy
Elapsed time (ave) 3.12235717773 [s]
Elapsed time (std) 0.61017062996 [s]
# Cython
Elapsed time (ave) 2.09492807388 [s]
Elapsed time (std) 0.304449698912 [s]
```

In both stats, cython is faster and more stable. Note that the concatenation of results comming from each thread dominates in time. [main_multi_thread04.py](./main_multi_thread04.py) shows that if I comment the *Y = np.concatenate(Y)*.

### Codes
- [multi_thread03.pyx](./multi_thread03.pyx)
- [main_multi_thread03.py](./main_multi_thread03.py)
- [main_multi_thread04.py](./main_multi_thread04.py)

## Conclusion

Seeing results among the experients, the result of numpy were fluctuating even if using the same code while Cython codes gradually speed up according the step-by-step optimization; however the results reported above may change in time and environement. Note also *with nogil* block we can not use python objects incliuding, creation, handling, and returning.

*Cython + threading + nogil* is a best choise based on my experience in some viewpoints, e.g., easiness to start using and to write,  memory efficiency, computational speed. Now I just ran the not-too-heavy computational tasks, so combination of cython, threadnig, and nogil, will have more benefit if executing much heavier tasks which can be parallel. On the other hand, for numpy, there are some room to speed up, i.e., using MKL. Thus, I should have compare with numpy w/ MKL.


# References
- https://lbolla.info/blog/2013/12/23/python-threads-cython-gil
- http://nealhughes.net/parallelcomp2/
- http://cython.readthedocs.io/en/latest/src/userguide/parallelism.html
- http://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html




