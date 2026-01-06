import numpy as np
import scipy.sparse.linalg
from pymanopt.manifolds import Grassmann
from pymanopt import Problem
# from pymanopt.solvers import TrustRegions

def rasr_nonlinear_eigenspace(L, k, alpha=1):
    n = L.shape[0]
    assert L.shape[1] == n, 'L must be square.'

    # Grassmann manifold description
    Gr = Grassmann(n, k)
    problem = Problem(manifold=Gr)

    # Cost function evaluation
    def cost(X):
        rhoX = np.sum(X**2, axis=1)
        return 0.5 * np.trace(X.T @ (L @ X)) + (alpha / 4) * rhoX.T @ np.linalg.solve(L, rhoX)

    # Euclidean gradient evaluation
    def egrad(X):
        rhoX = np.sum(X**2, axis=1)
        return L @ X + alpha * np.diag(np.linalg.solve(L, rhoX)) @ X

    # Euclidean Hessian evaluation
    def ehess(X, U):
        rhoX = np.sum(X**2, axis=1)
        rhoXdot = 2 * np.sum(X * U, axis=1)
        return L @ U + alpha * np.diag(np.linalg.solve(L, rhoXdot)) @ X + alpha * np.diag(np.linalg.solve(L, rhoX)) @ U

    problem.cost = cost
    problem.egrad = egrad
    problem.ehess = ehess

    # Initialization
    np.random.seed(0)
    X = np.random.randn(n, k)
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    X = U @ Vt
    vals, U0 = scipy.sparse.linalg.eigs(L + alpha * np.diag(np.linalg.solve(L, np.sum(X**2, axis=1))), k=k)
    X0 = U0

    # Call the solver
    solver = TrustRegions()
    Xsol = solver.solve(problem, X0)

    return Xsol, vals
