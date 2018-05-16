from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.admm import cbpdndl
from sporco import util
import sporco.metric as sm
from sporco.admm import cbpdn

import sys
import math
#from tensorflow.examples.tutorials.mnist import input_data
from sporco import plot


# Load dataset
#mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
#digits = mnist.train.images

exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
S = np.dstack((S1, S2, S3, S4, S5))

npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)

# Create initial dictionary
np.random.seed(12345)
D0 = np.random.randn(8, 8, 64)

# Set regularization parameter and options for dictionary learning solver 
lmbda = 0.2
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            method='cns')

# Create the solver and solve
d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt, method='cns')
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

#D1 = D1.squeeze()
#fig = plot.figure(figsize=(14, 7))
#plot.subplot(1, 2, 1)
#plot.imview(util.tiledict(D0), fig, 'D0')
#plot.subplot(1, 2, 2)
#plot.imview(util.tiledict(D1), fig, 'D1')
#fig.show()


# Sparse coding step

# Load image
img = util.ExampleImages().image('kodim23.png', scaled=True, gray=True,
                                idxexp=np.s_[160:416,60:316])
npd = 16
fltlmbd = 10
sl, sh = util.tikhonov_filter(img, fltlmbd, npd)

# Set solver options
lmbda = 5e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                              'RelStopTol': 5e-3, 'AuxVarObj': False})
# Initialise and run CSC solver
b = cbpdn.ConvBPDN(D1, sh, lmbda, opt, dimK=0)
X = b.solve()
print("ConvBPDN solve time: %.2fs" % b.timer.elapsed('solve'))


# reconstruct image from sparse representation
shr = b.reconstruct().squeeze()
imgr = sl + shr
print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(img, imgr))


fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(sl, 'Lowpass component', fig)
plot.subplot(1, 2, 2)
plot.imview(np.sum(abs(X), axis=b.cri.axisM).squeeze(), plot.cm.Blues,
            'Sparse representation', fig)
fig.show()

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(img, 'Original', fig)
plot.subplot(1, 2, 2)
plot.imview(imgr, 'Reconstructed', fig)
fig.show()

its = b.getitstat()
fig = plot.figure((20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, fig, 'Iterations', 'Functional')
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, fig,
          'semilogy', 'Iterations', 'Residual',
          ['Primal', 'Dual'])
plot.subplot(1, 3, 3)
plot.plot(its.Rho, fig, 'Iterations', 'Penalty Parameter')
fig.show()
