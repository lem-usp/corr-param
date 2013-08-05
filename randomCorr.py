import numpy as np

def rand_corr(n, ke):
    """Return a n x n random correlation matrix"""
    c = np.zeros((n,n))
    b = np.tri(n, n)

    c[1:n, 0] = -1 + 2*np.round(np.random.rand(1, n-1)*10**8)/10**8
    b[1:n, 0] = c[1:n, 0]

    for i in xrange(1, n):
        b[i, 1:i+1] = np.sqrt(1 - c[i, 0]**2)

    for i in xrange(2, n):
        for j in xrange(1, i):
            b1 = np.dot(b[j, 0:j], b[i, 0:j].T)
            b2 = np.dot(b[j, j], b[i, j])
            z = b1 + b2
            y = b1 - b2
            if b2 < ke:
                c[i, j] = b1
                cosinv = 0
            else:
                c[i, j] = y + (z - y)*np.round(np.random.rand()*10**8)/10**8

            cosinv = (c[i, j] - b1)/b2

            if np.isfinite(cosinv):
                if cosinv > 1:
                    b[i, j] = b[i, j]
                    b[i, j+1:n+1] = 0
                elif cosinv < -1:
                    b[i, j] = -b[i, j]
                    b[i, j+1:n+1] = 0
                else:
                    b[i, j] = b[i, j]*cosinv
                    sinTheta = np.sqrt(1 - cosinv**2)
                    for k in xrange(j+1, n):
                        b[i, k] = b[i, k]*sinTheta

    c = c + c.T + np.eye(n)
    perm = np.random.permutation(n)
    c = (c[perm,])[:,perm]

    return c

def triang_decomp(c):
    """Return a lower triangular matrix B that B * B.T = C"""
    n = c.shape[0]
    b = np.tri(n, n)

    b[1:n, 0] = c[1:n, 0]

    for i in xrange(1, n):
        b[i, 1:i+1] = np.sqrt(1 - c[i, 0]**2)

    for i in xrange(2, n):
        for j in xrange(1, i):
            b1 = np.dot(b[j, 0:j], b[i, 0:j].T)
            b2 = np.dot(b[j, j], b[i, j])
            cosinv = (c[i, j] - b1)/b2

            if np.isfinite(cosinv):
                if cosinv > 1:
                    b[i, j] = b[i, j]
                    b[i, j+1:n+1] = 0
                elif cosinv < -1:
                    b[i, j] = -b[i, j]
                    b[i, j+1:n+1] = 0
                else:
                    b[i, j] = b[i, j]*cosinv
                    sinTheta = np.sqrt(1 - cosinv**2)
                    for k in xrange(j+1, n):
                        b[i, k] = b[i, k]*sinTheta

    return b

def calc_params(b):
    """ Given a lower trianguler B matrix returned by triang_decomp, return angle parameters"""
    n = b.shape[0]
    p = np.zeros(n*n).reshape((n, n))

    for j in xrange(n-1):
        for i in xrange(j, n):
            p[i, j] = np.arccos(b[i, j]/np.exp(np.sum(np.log(np.sin(p[i, 0:j])))))

    p[np.isnan(p)] = 0

    return p

def triang_from_params(p):
    """Given a p param matrix, calculate the lower triangular B matrix"""
    n = p.shape[0]
    b = np.zeros(n*n).reshape((n,n))

    for j in xrange(n-1):
        for i in xrange(n):
            b[i, j] = np.cos(p[i,j])*np.product(np.sin(p[i, 0:j]))

    b[n-1,n-1] = np.product(np.sin(p[n-1, 0:n-1]))

    return b
