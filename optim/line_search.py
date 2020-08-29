import numpy
import scipy
import warnings

def armijo(loss_fun, grad, x, d, alpha = 1.0, c = 0.1, delta = 0.5, max_iter = 10):
    """
    line search using armijo's condition
    :param loss_fun: Loss function
    :param grad: gradient at x
    :param x: current point x
    :param d: search direction (e.g. NEGATIVE gradient)
    :param alpha_init: initial alpha
    :param c: Slope parameter for line of sufficient decrease.
    :param delta: Backtracking multiplier
    :param max_iter: maximum iteration
    :return: alpha found when Armijo rule is met or max_iter is reached
    """

    fx = loss_fun(x)
    for iter in range(max_iter):
        if loss_fun(x + alpha * d) <= fx + c * alpha * (d @ grad):
            return alpha
        else:
            alpha = alpha * delta

    warnings.warn("Maximum iteration for Armijo line search is reached.")
    return alpha
