'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    x = inv(A.T @ A) @ A.T @ b
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    lu = splu(A.T @ A, permc_spec="NATURAL")
    x = lu.solve(A.T @ b)
    U = lu.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutation_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    lu = splu(A.T @ A, permc_spec="COLAMD")
    x = lu.solve(A.T @ b)
    U = lu.U
    return x, U

def solve_lu_custom(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition. Custom forward/backward substitution
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html

    lu = splu(A.T @ A, permc_spec="NATURAL")
    L = lu.L
    U = lu.U

    y = solve_triangular_custom(L, A.T @ b, lower=True)
    x = solve_triangular_custom(U, y, lower=False)

    return x, U

def solve_triangular_custom(T, b, lower=True):
    # Custom implementation according to https://github.com/scipy/scipy/blob/v1.15.2/scipy/sparse/linalg/_dsolve/linsolve.py#L0-L1

    n = T.shape[0]
    x = np.zeros(n)
    T = T.tocsr()

    if lower:

        # Forward
        for i in range(n):
            row_start = T.indptr[i]
            row_end = T.indptr[i+1]
            total = b[i]
            diag = None

            for idx in range(row_start, row_end):
                j = T.indices[idx]
                val = T.data[idx]

                if j == i:
                    diag = val
                elif j < i:
                    total -= val * x[j]
                elif j > i:
                    break
            x[i] = total / diag

    else:

        # Backward
        for i in reversed(range(n)):
            row_start = T.indptr[i]
            row_end = T.indptr[i + 1]
            total = b[i]
            diag = None

            for idx in range(row_start, row_end):
                j = T.indices[idx]
                val = T.data[idx]

                if j == i:
                    diag = val
                elif j > i:
                    total -= val * x[j]

            x[i] = total / diag
    
    return x


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    z, R, E, rank = rz(A, b, permc_spec="NATURAL")
    x = spsolve_triangular(R, z, lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    z, R, E, rank = rz(A, b, permc_spec="COLAMD")
    x = permutation_vector_to_matrix(E) @ spsolve_triangular(R, z, lower=False)
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matrix
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
        'lu_custom': solve_lu_custom,
    }

    return fn_map[method](A, b)
