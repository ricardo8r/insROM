import sys
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from scipy import signal #for sSVD
from SVD import SVD
import matplotlib.pyplot as plt

def rSpace(method,typeb,library,npar):
    algorithm = {"POD":POD(typeb,library,npar),"DMD":DMD(typeb,library,npar),"TD":TD(typeb,library,npar)}
    return algorithm[method]

class POD:
    
    def __init__(self,typeb,library,npar):
        self.svd = SVD(typeb,library)
        self.npar =npar

    def setData(self,X,r):
        if self.npar == 1:
            self.svd.setData(X[:,:,0],r)
        else:
            self.svd.setData(tl.unfold(tl.tensor(np.swapaxes(X,1,2)),0),r)
        self.r = r

    def create(self):
        self.U, self.S, self.V = self.svd.decompose()

    def filter(self):
        self.U, self.S, self.V, self.r = filterCor(self.U,self.S,self.V)

class DMD:
    
    def __init__(self,typeb,library,npar):
        self.svd = SVD(typeb,library)
        self.npar =npar

    def setData(self,X,r):
        if self.npar == 1:
            self.X = X[:,:,0]
        else:
            self.X = tl.unfold(np.swapaxes(X,1,2),0)
        self.r = r

    def create(self):
        X1 = self.X[:,0:self.X.shape[1]-1]
        X2 = self.X[:,1:self.X.shape[1]]

        self.svd.setData(X1,self.r)
        U1, S1, V1 = self.svd.decompose()
        Sb = U1.T @ X2 @ V1
        Sb = Sb/S1

        self.svd.setData(Sb,self.r)
        U2, self.S, V2 = self.svd.decompose()
        self.U = U1 @ U2
        self.V = V1 @ V2

    def filter(self):
        self.U, self.S, self.V, self.r = filterCor(self.U,self.S,self.V)
    
class TD:
    
    def __init__(self,typeb,library,npar):
        self.typeb = typeb
        self.library = library
        self.svd = SVD(typeb,library)
        self.npar =npar

    def setData(self,X,r):
        self.X = tl.tensor(np.swapaxes(X,0,2))
        self.r = r

    def create(self):
        #U = []
        #S = []
        #V = []
        #X = []
        #X.append(tl.unfold(self.X,0))
        #X.append(tl.unfold(self.X,1))
        #X.append(tl.unfold(self.X,2))
        #for iax in range(self.X.ndim):
        #    self.svd.setData(X[iax],self.r)
        #    U1, S1, V1 = self.svd.decompose()
        #    U.append(U1)
        #    S.append(S1)
        #    V.append(V1)
        #S,U = tucker(self.X,ranks=[10,25,25])
        #self.S=np.diag(np.abs(S[:,0,:]))
        #self.U = U[2]
        #self.S = S[2]
        #self.V = V[2]
        #print(U[0].shape,U[1].shape,U[2].shape,S.shape)
        self.svd.setData(tl.unfold(self.X,2),self.r)
        self.U, self.S, self.V = self.svd.decompose()

    def filter(self):
        self.U, self.S, self.V, self.r = filterCor(self.U,self.S,self.V)
        

def filterCor(U,S,V):
    noise = []
    peak_p = 0
    for iip in range(V.shape[1]):
        corr = signal.correlate(V[1:V.shape[1]-1,iip],V[1:V.shape[1]-1,iip])
        peak = signal.find_peaks(corr)
        if (len(peak[0])< peak_p): 
            noise.append(iip)
        else:
            peak_p = len(peak[0])

    Ut = np.delete(U,noise,axis=1)
    Vt = np.delete(V,noise,axis=1)
    St = np.delete(S,noise,axis=0)
    rt = Vt.shape[1]
    return Ut, St, Vt, rt

def bPlot(basis):
    plt.plot(basis.S)
    plt.xlabel('r') 
    plt.ylabel('Energy')
    plt.savefig('Basis.png')
