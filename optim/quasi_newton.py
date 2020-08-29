import numpy
import scipy
from . import line_search

def DFP(loss_fun, grad_fun, x_init, alpha = 1.0, tol = 1e-6, max_iter = 100):
    """

    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param hess_fun: hessian function
    :param x_init: initial x
    :param max_iter: maximum iteration
    :param lm: Levenberg–Marquardt
    :return:
    """
    n = x_init.shape[0]
    rec = numpy.empty([max_iter + 1, n])
    rec[:] = numpy.nan
    x = x_init.copy()

    for i in range(max_iter):
        rec[i, :] = x
        print(i, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))
        if i == 0:
            H = numpy.eye(n)
        else:
            delta_grad = grad - prev_grad
            temp = delta_grad @ H.T
            H += numpy.outer(delta_x, delta_x) / numpy.inner(delta_x, delta_grad) - \
                temp.T @ temp / (delta_grad @ H @ numpy.reshape(delta_grad, [-1, 1]))

        prev_grad = grad

        d = -H @ grad
        delta_x = line_search.armijo(loss_fun, grad, x, d, alpha=alpha) * d
        x = x + delta_x
        if (delta_x ** 2).sum() < tol:
            break

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec


def BFGS(loss_fun, grad_fun, x_init, alpha = 1.0, tol = 1e-6, max_iter = 100):
    """

    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param hess_fun: hessian function
    :param x_init: initial x
    :param max_iter: maximum iteration
    :param lm: Levenberg–Marquardt
    :return:
    """
    n = x_init.shape[0]
    rec = numpy.empty([max_iter + 1, n])
    rec[:] = numpy.nan
    x = x_init.copy()

    for i in range(max_iter):
        rec[i, :] = x
        print(i, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))
        if i == 0:
            H = numpy.eye(n)
        else:
            delta_grad = grad - prev_grad
            temp1 = numpy.inner(delta_x, delta_grad)
            temp2 = numpy.eye(n) - numpy.outer(delta_x, delta_grad) / temp1
            H = numpy.outer(delta_x, delta_x) / temp1 + temp2 @ H @ temp2.T

        prev_grad = grad

        d = -H @ grad
        delta_x = line_search.armijo(loss_fun, grad, x, d, alpha=alpha) * d
        x = x + delta_x
        if (delta_x ** 2).sum() < tol:
            break

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec