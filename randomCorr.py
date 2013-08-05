
import numpy as np

def mkRandCorr(n, ke):
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
