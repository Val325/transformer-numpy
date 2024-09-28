import math
import numpy as np

"""
def layer_norm(x):
    average = x.mean()
    variance = x.var()
    return (x - average)/(np.sqrt(variance))

def layer_norm_derivative(x):
    pass
"""

# https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
# batch and layer norm has similar formula
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
class layer_norm():
    def __init__(self, eps):
        self.w = None
        self.b = None
        self.gamma = None
        self.beta = None
        self.eps_save = eps
    def feedforward(self, x, gamma, beta, eps):
        N = len(x) 
        
        #step1: calculate mean
        mu = 1./N * np.sum(x)

        #step2: subtract mean vector of every trainings example
        xmu = x - mu

        #step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        #step4: calculate variance
        var = 1./N * np.sum(sq)

        #step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + eps)

        #step6: invert sqrtwar
        ivar = 1./sqrtvar

        #step7: execute normalization
        xhat = xmu * ivar

        #step8: Nor the two transformation steps
        gammax = gamma * xhat

        #step9
        out = gammax + beta

        #store intermediate
        #cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
        self.xhat_save = xhat
        self.gamma_save = gamma
        self.xmu_save = xmu
        self.ivar_save = ivar
        self.sqrtvar_save = sqrtvar
        self.var_save = var
        self.eps_save = eps

        return out
    def backforward(self, dout):
        #unfold the variables stored in cache
        #xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
        xhat = self.xhat_save
        gamma = self.gamma_save
        xmu = self.xmu_save
        ivar = self.ivar_save
        sqrtvar = self.sqrtvar_save
        var = self.var_save
        eps = self.eps_save

        #get size 
        N = len(dout)

        #step9
        dbeta = np.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable

        #step8
        dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma

        #step7
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        #step6
        dsqrtvar = -1. /(sqrtvar**2) * divar

        #step5
        dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

        #step4
        dsq = 1. /N * np.ones(N) * dvar

        #step3
        dxmu2 = 2 * xmu * dsq

        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

        #step1
        dx2 = 1. /N * np.ones(N) * dmu

        #step0
        dx = dx1 + dx2

        return dx, dgamma, dbeta

#array = np.array([0.1, 0.4, 0.5, 0.3])
#layer_n = layer_norm(0.0001)
#feed = layer_n.feedforward(array, 1.1, 1, 0.0001)
#backprop = layer_n.backforward(feed) 
#print(feed)
#print(backprop)
