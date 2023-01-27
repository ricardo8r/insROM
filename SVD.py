import sys,petsc4py,slepc4py
from mpi4py import MPI
import numpy as np
#import pywt  #to do: wavelet transforms
from scipy import signal #for sSVD
from sklearn.decomposition import TruncatedSVD,PCA
from petsc4py import PETSc
from slepc4py import SLEPc

def SVD(typeb,library):
    method = {"rSVD":rSVD(library),"SVD":cSVD(library),"sSVD":sSVD(library)}
    return method[typeb]

# Randomized SVD
class rSVD: 

    def __init__(self,library):
        self.library = library

    def setData(self,X,r):
        self.X = X
        self.r = r
        self.q = 1
        self.p = np.int(self.r*0.2)

    def decompose(self):
        if(self.library=='numpy'):
            ny = self.X.shape[1]
            P = np.random.randn(ny,self.r+self.p)
            Z = self.X @ P
            for k in range(self.q):
                Z = self.X @ (self.X.T @ Z)
            
            Q, R = np.linalg.qr(Z,mode='reduced')
    
            Y = Q.T @ self.X
            svd = cSVD(self.library)
            svd.setData(Y,self.X.shape[1])
            UY, S, V = svd.decompose()
            U = Q @ UY

            return U[:,:self.r], S[:self.r], V[:,:self.r]

        elif(self.library=='scikit'):
            svd = TruncatedSVD(n_components=self.r,n_iter=self.q)
            #svd = PCA(n_components=self.r,svd_solver='randomized',iterated_power=self.q)
            #svd = IncrementalPCA(n_components=self.r,batch_size=500)
            svd.fit(self.X.T)
            U = svd.components_.T
            S = svd.singular_values_
            V = np.diag(S) @ U.T @ self.X

            return U[:,:self.r], S[:self.r], V.T[:,:self.r]

    
    # SVD
class cSVD: 

    def __init__(self,library):
        self.library = library

    def setData(self,X,r):
        self.X = X
        self.r = r

    def decompose(self):
    
        if(self.library=='numpy'):
    
            U, S, VT = np.linalg.svd(self.X,full_matrices=0)
            return U[:,:self.r], S[:self.r], VT.T[:,:self.r]

        elif(self.library=='scikit'):
            svd = TruncatedSVD(n_components=self.r,algorithm='arpack')
            #svd = PCA(n_components=self.r,svd_solver='arpack')
            #svd = IncrementalPCA(n_components=self.r)
            svd.fit(self.X.T)
            U = svd.components_.T
            S = svd.singular_values_
            V = np.diag(S) @ U.T @ self.X

            return U[:,:self.r], S[:self.r], V.T[:,:self.r]

        elif(self.library=='slepc'):
            A = PETSc.Mat().create(MPI.COMM_WORLD)
            A.setType(PETSc.Mat.Type.DENSE)
            A.setSizes(self.X.shape)
            A.setFromOptions()
            A.setUp()
            for i in range(self.X.shape[0]):
                for j in range(self.X.shape[1]):
                    A.setValue(i,j,self.X[i,j])
            A.assemble()
    
            svd = SLEPc.SVD()
            svd.create(MPI.COMM_WORLD)
            svd.setOperator(A)
            svd.setType(svd.Type.TRLANCZOS)
            svd.setFromOptions()
            svd.setDimensions(self.r)
            svd.solve()
    
            nconv = svd.getConverged()
            S = np.zeros((nconv))
            U = np.zeros((self.X.shape[0],nconv))
            VT = np.zeros((nconv,self.X.shape[1]))
            if nconv > 0:
                VTs, Us = A.getVecs()
                for i in range(nconv):
                    S[i] = svd.getSingularTriplet(i,Us,VTs)
                    for j in range(self.X.shape[0]):
                        U[j,i] = np.real(Us.getValue(j))
                    for j in range(nconv):
                        VT[j,i] = np.real(VTs.getValue(j))
    
            return U, S, VT.T

    # sSVD
class sSVD: 

    def __init__(self,library):
        self.library = library

    def setData(self,X,r):
        self.X = X
        self.r = r

    def decompose(self):
        fX,pdX = signal.welch(self.X,fs=10.0,window='hann',nperseg=self.X.shape[1],noverlap=None,nfft=None,detrend='constant',return_onesided=True,scaling='density',axis=-1,average='median')
        svd = cSVD(self.library)
        svd.setData(pdX,self.r)
        U, S, V = svd.decompose()
    
        return U, S, V
