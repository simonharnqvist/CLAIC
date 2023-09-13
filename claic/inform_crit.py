import numpy as np
from matrices import jacobian, hessian


def claic(f, likelihood, estimates):
    """Calculate composite likelihood AIC.

    Args:
        f (function): Composite likelihood function.
        likelihood (float): Likelihood estimate.
        estimates (tuple): Parameter estimates.

    Returns:
        float: Composite likelihood AIC.
    """

    jacobian_mat = jacobian(f, estimates)
    hessian_mat = hessian(f, estimates)

    cl_akaike = (-2 * likelihood 
                 + 2 * np.trace(jacobian_mat @ np.linalg.inv(hessian_mat)))

    return cl_akaike


def clbic(f, likelihood, estimates, n_samples):
    """Calculate composite likelihood BIC.

    Args:
        f (function): Composite likelihood function.
        likelihood (float): Likelihood estimate.
        estimates (tuple): Parameter estimates.
        n_samples (int): Sample size.

    Returns:
        float: Composite likelihood BIC.
    """

    jacobian_mat = jacobian(f, estimates)
    hessian_mat = hessian(f, estimates)

    cl_bayes = (-2 * likelihood
                + np.log(n_samples)
                * np.trace(jacobian_mat @ np.linalg.inv(hessian_mat)))

    return cl_bayes
