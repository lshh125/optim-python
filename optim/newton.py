import numpy
import scipy

def newton(loss_fun, grad_fun, hess_fun, x_init, max_iter = 100, lm=0):
    """

    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param hess_fun: hessian function
    :param x_init: initial x
    :param max_iter: maximum iteration
    :param lm: Levenberg–Marquardt
    :return:
    """
    rec = numpy.empty([max_iter + 1, x_init.shape[0]])
    rec[:] = numpy.nan
    x = x_init.copy()
    for i in range(max_iter):
        rec[i, :] = x
        print(i, x, loss_fun(x))
        d = -numpy.squeeze(scipy.linalg.solve(hess_fun(x) + numpy.eye(x_init.shape[0]) * lm, numpy.squeeze(grad_fun(x)), assume_a="sym"))
        x = x + d

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec