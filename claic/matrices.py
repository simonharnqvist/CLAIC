import numpy as np
import numdifftools as nd


def hessian(f, parameters):
    """
    Get Hessian matrix for a function f at a given set of parameter values.
    """
    hess_f = nd.core.Hessian(f)
    return hess_f(np.array(parameters))

def jacobian(f, parameters):
    """
    Get Jacobian matrix for a function f at a given set of parameter values.
    """
    jac_f = nd.core.Jacobian(f)
    return jac_f(np.array(parameters))