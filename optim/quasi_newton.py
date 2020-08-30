import numpy
import scipy
from . import line_search
from itertools import chain
from .utils import reversed_range

def dfp(loss_fun, grad_fun, x_init, alpha = 1.0, tol = 1e-6, max_iter = 100):
    """

    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param x_init: initial x
    :param alpha: maximum step size for line search
    :param tol: criterion of stop
    :param max_iter: maximum iteration
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


def bfgs(loss_fun, grad_fun, x_init, alpha = 1.0, tol = 1e-6, max_iter = 100):
    """
    BFGS with initial B^(-1) set to (y^T s) / (y^T y) I
    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param x_init: initial x
    :param alpha: maximum step size for line search
    :param tol: criterion of stop
    :param max_iter: maximum iteration
    :return:
    """
    n = x_init.shape[0]
    rec = numpy.empty([max_iter + 1, n])
    rec[:] = numpy.nan
    x = x_init.copy()

    for iter in range(max_iter):
        rec[iter, :] = x
        print(iter, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))
        if iter == 0:
            H = numpy.eye(n)
        else:
            delta_grad = grad - prev_grad
            if iter == 1:
                H = numpy.eye(n) * numpy.inner(delta_grad, delta_x) / numpy.inner(delta_grad, delta_grad)
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


def bfgs0(loss_fun, grad_fun, x_init, alpha = 1.0, tol = 1e-6, max_iter = 100):
    """
    BFGS with initial B^(-1) set to I
    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param x_init: initial x
    :param alpha: maximum step size for line search
    :param tol: criterion of stop
    :param max_iter: maximum iteration
    :return:
    """
    n = x_init.shape[0]
    rec = numpy.empty([max_iter + 1, n])
    rec[:] = numpy.nan
    x = x_init.copy()

    for iter in range(max_iter):
        rec[iter, :] = x
        print(iter, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))
        if iter == 0:
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


def l_bfgs(loss_fun, grad_fun, x_init, m = 5, max_step = 1.0, tol = 1e-6, max_iter = 100):
    """
    L-BFGS
    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param x_init: initial x
    :param max_step: maximum step size for line search
    :param tol: criterion of stop
    :param max_iter: maximum iteration
    :return:
    """
    n = x_init.shape[0]
    rec = numpy.empty([max_iter + 1, n])
    rec[:] = numpy.nan
    x = x_init.copy()

    delta_x = numpy.empty([m, n])
    delta_grad = numpy.empty([m, n])
    rho = numpy.empty(m)
    alpha = numpy.empty(m)
    beta = numpy.empty(m)

    for iter in range(max_iter):
        iter_mod_m = iter % m

        rec[iter, :] = x
        print(iter, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))

        if iter == 0:
            d = -grad
        else:
            iter_1_mod_m = (iter - 1) % m
            delta_grad[iter_1_mod_m, :] = grad - prev_grad
            rho[iter_1_mod_m] = 1 / numpy.inner(delta_grad[iter_1_mod_m, :], delta_x[iter_1_mod_m, :])
            # two-loop recursion
            iter_1 = iter - 1
            range1 = reversed_range(iter_mod_m) if iter < m else chain(reversed_range(iter_mod_m), range(m, iter_mod_m))
            range2 = range(iter_mod_m) if iter < m else chain(range(iter_mod_m, ))
            q = grad
            for i in range1:
                alpha[i] = rho[i] * numpy.inner(delta_x[i, :], q)
                q = q - alpha[i] * delta_grad[i, :]
            d = numpy.inner(delta_x[iter_1_mod_m, :], delta_grad[iter_1_mod_m, :]) / numpy.inner(delta_grad[iter_1_mod_m, :], delta_grad[iter_1_mod_m, :]) * q
            for i in range2:
                beta[i] = rho[i] * numpy.inner(delta_grad[i, :], d)
                d = d + (alpha[i] - beta[i]) * delta_x[i, :]
            d = -d

        prev_grad = grad

        #print(x, d)
        delta_x[iter_mod_m, :] = line_search.armijo(loss_fun, grad, x, d, alpha=max_step) * d
        x = x + delta_x[iter_mod_m, :]
        if (delta_x[iter_mod_m, :] ** 2).sum() < tol:
            break

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec

def l_bfgs0(loss_fun, grad_fun, x_init, m = 5, max_step = 1.0, tol = 1e-6, max_iter = 100):
    """
    L_BFGS with initial B^(-1) set to I
    :param loss_fun: loss function
    :param grad_fun: gradient funciton
    :param x_init: initial x
    :param max_step: maximum step size for line search
    :param tol: criterion of stop
    :param max_iter: maximum iteration
    :return:
    """
    n = x_init.shape[0]
    rec = numpy.empty([max_iter + 1, n])
    rec[:] = numpy.nan
    x = x_init.copy()

    delta_x = numpy.empty([m, n])
    delta_grad = numpy.empty([m, n])
    rho = numpy.empty(m)
    alpha = numpy.empty(m)
    beta = numpy.empty(m)

    for iter in range(max_iter):
        iter_mod_m = iter % m

        rec[iter, :] = x
        print(iter, x, loss_fun(x))
        grad = numpy.squeeze(grad_fun(x))

        if iter == 0:
            d = -grad
        else:
            iter_1_mod_m = (iter - 1) % m
            delta_grad[iter_1_mod_m, :] = grad - prev_grad
            rho[iter_1_mod_m] = 1 / numpy.inner(delta_grad[iter_1_mod_m, :], delta_x[iter_1_mod_m, :])
            # two-loop recursion
            iter_1 = iter - 1
            range1 = reversed_range(iter_mod_m) if iter < m else chain(reversed_range(iter_mod_m), range(m, iter_mod_m))
            range2 = range(iter_mod_m) if iter < m else chain(range(iter_mod_m, ))
            q = grad
            for i in range1:
                alpha[i] = rho[i] * numpy.inner(delta_x[i, :], q)
                q = q - alpha[i] * delta_grad[i, :]
            #d = numpy.inner(delta_x[iter_1_mod_m, :], delta_grad[iter_1_mod_m, :]) / numpy.inner(delta_grad[iter_1_mod_m, :], delta_grad[iter_1_mod_m, :]) * q
            d = q
            for i in range2:
                beta[i] = rho[i] * numpy.inner(delta_grad[i, :], d)
                d = d + (alpha[i] - beta[i]) * delta_x[i, :]
            d = -d

        prev_grad = grad

        #print(x, d)
        delta_x[iter_mod_m, :] = line_search.armijo(loss_fun, grad, x, d, alpha=max_step) * d
        x = x + delta_x[iter_mod_m, :]
        if (delta_x[iter_mod_m, :] ** 2).sum() < tol:
            break

    rec[max_iter, :] = x
    print(max_iter, x, loss_fun(x))

    return rec