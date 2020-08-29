import numpy
from . import line_search

def gradient_descent(loss_fun, grad_fun, x_init, lr = 0.01, max_iter = 100):
    rec = numpy.empty([max_iter + 1, x_init.shape[0]])
    rec[:] = numpy.nan
    x = x_init.copy()
    for i in range(max_iter):
        rec[i, :] = x
        print(i, x, loss_fun(x))
        d = -numpy.squeeze(grad_fun(x))
        x = x + lr * d

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec

def steepest_descent_with_armijo(loss_fun, grad_fun, x_init, alpha = 1., max_iter = 100):
    rec = numpy.empty([max_iter + 1, x_init.shape[0]])
    rec[:] = numpy.nan
    x = x_init.copy()
    for i in range(max_iter):
        rec[i, :] = x
        print(i, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))
        d = -grad
        x = x + line_search.armijo(loss_fun, grad, x, d, alpha, ) * d

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec