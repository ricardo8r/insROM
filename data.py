import os,sys,getopt
import re
from mpi4py import MPI
import h5py
import numpy as np

class reader:
    
    def __init__(self,case,problem,npar):
        self.case = case
        self.problem = problem
        self.npar = npar

        # Names of fields depending on problem
        if(self.problem=='Navier-Stokes'):
            self.fields = ['Velocity X','Velocity Y','Pressure']
        elif(self.problem=='Boussinesq'):
            self.fields = ['Velocity X','Velocity Y','Pressure','Temperature']
        elif(self.problem=='Zero-Mach'):
            self.fields = ['Velocity X','Velocity Y','Temperature','Pressure']
        elif(self.problem=='Helmholtz'):
            self.fields = ['Helmholtz']
        elif(self.problem=='Maxwell'):
            self.fields = ['Maxwell X','Maxwell Y']
        else:
            self.fields = str.split(problem,',')
        
        self.nf = len(self.fields)

    def readMesh(self):
        snp_file = h5py.File(str(self.case)+'.msh','r',driver='mpio',comm=MPI.COMM_WORLD)
        snp_file.atomic = True
    
        self.nn =  len(snp_file['Mass'])
        self.LMass = np.zeros((self.nf*self.nn))
        self.iLMass = np.zeros((self.nf*self.nn))
        for iif in range(self.nf):
            self.iLMass[iif:self.nn*self.nf:self.nf] = np.sqrt(snp_file['Mass'])
        snp_file.close()
        self.LMass = 1.0/self.iLMass

    def readMesh2(self,mesh_files):
        snp_file = h5py.File(mesh_files[0],'r',driver='mpio',comm=MPI.COMM_WORLD)
        #snp_file = h5py.File(mesh_files[0],'r')
        snp_file.atomic = True
    
        self.nn =  len(snp_file['Mass'])
        self.LMass = np.zeros((self.nf*self.nn))
        self.iLMass = np.zeros((self.nf*self.nn))
        for iif in range(self.nf):
            self.iLMass[iif:self.nn*self.nf:self.nf] = np.sqrt(snp_file['Mass'])
        snp_file.close()
        self.LMass = 1.0/self.iLMass

    def readSnaps(self):
        snp_file = h5py.File(str(self.case)+'.rom.snp','r',driver='mpio',comm=MPI.COMM_WORLD)
        snp_file.atomic = True
    
        snp = snp_file['Snapshots']
        self.ns = snp.attrs['NSnaps']
        
        self.X = np.zeros((self.nf*self.nn,self.ns,self.npar))
        iif = 0
        for field in self.fields:
            iis = 0
            self.X.T[:,:,iif:self.nn*self.nf:self.nf] = snp_file['Snapshots/'+str(field)]
            iif+= 1
        snp_file.close()

    def readSnaps2(self,files):
        snp_file = h5py.File(files[0],'r',driver='mpio',comm=MPI.COMM_WORLD)
        #snp_file = h5py.File(files[0],'r')
        snp_file.atomic = True
    
        snp = snp_file['Snapshots']
        self.ns = snp.attrs['NSnaps']
        self.X = np.zeros((self.nf*self.nn,self.ns,self.npar))
        snp_file.close()

        iip = 0
        for file in files:
            snp_file = h5py.File(file,'r',driver='mpio',comm=MPI.COMM_WORLD)
            #snp_file = h5py.File(file,'r')
            snp_file.atomic = True
            iif = 0
            for field in self.fields:
                iis = 0
                self.X.T[iip,:,iif:self.nn*self.nf:self.nf] = snp_file['Snapshots/'+str(field)]
                iif+= 1
            iip+= 1
            snp_file.close()

    def setMean(self):
        self.XMean = np.zeros((self.nn*self.nf,self.npar))
        self.XMean = np.average(self.X,axis=1)
    
    def center(self):
        for iip in range(self.npar):
            for iis in range(self.ns):
                self.X[:,iis,iip] = self.LMass * (self.X[:,iis,iip] - self.XMean[:,iip])

