
import numpy as np
import utils
X, T = np.loadtxt('data_X.txt'), np.loadtxt('data_T.txt')
utils.digit(X.reshape(1,12,28,28).transpose(0,2,1,3).reshape(28,12*28),9,0.75)

W, B = utils.loadparams()
L = len(W)

import numpy
A = [X]+[None]*L
for l in range(L):
    A[l+1] = numpy.maximum(0, A[l].dot(W[l])+B[l])

for i in range(3):
    utils.digit(X[i].reshape(28, 28), 0.75, 0.75)
    p = A[L][i]
    print("  ".join(['[%1d] %.1f'%(d, p[d]) for d in range(10)]))

R = [None]*L + [A[L]*(T[:,None]==numpy.arange(10))]


def rho(w,l):  return w + [None,0.1,0.0,0.0][l] * numpy.maximum(0,w)
def incr(z,l): return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9



for l in range(1,L)[::-1]:

    w = rho(W[l],l)
    b = rho(B[l],l)

    z = incr(A[l].dot(w)+b,l)    # step 1
    s = R[l+1] / z               # step 2
    c = s.dot(w.T)               # step 3
    R[l] = A[l]*c                # step 4


w  = W[0]
wp = numpy.maximum(0,w)
wm = numpy.minimum(0,w)
lb = A[0]*0-1
hb = A[0]*0+1

z = A[0].dot(w)-lb.dot(wp)-hb.dot(wm)+1e-9        # step 1
s = R[1]/z                                        # step 2
c,cp,cm  = s.dot(w.T),s.dot(wp.T),s.dot(wm.T)     # step 3
R[0] = A[0]*c-lb*cp-hb*cm                         # step 4


utils.digit(X.reshape(1,12,28,28).transpose(0,2,1,3).reshape(28,12*28),9,0.75)
utils.heatmap(R[0].reshape(1,12,28,28).transpose(0,2,1,3).reshape(28,12*28),9,0.75)

