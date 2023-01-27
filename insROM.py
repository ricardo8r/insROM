import os,sys,re
import argparse
from mpi4py import MPI
import h5py
import numpy as np
from data import reader
from rSpace import rSpace,bPlot
import time
from find_files import find_file

start_time = time.time()

rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

parser = argparse.ArgumentParser(description='Construct reduced space')
parser.add_argument('-c', dest='case', action='store',type=str,help='Case name')
parser.add_argument('-s', dest='sets', action='store',type=str,help='Single or multiple datasets')
parser.add_argument('-m', dest='method', action='store',type=str,help='Construction method (POD,DMD)')
parser.add_argument('-t', dest='typed', action='store',type=str,help='Decomposition type (rSVD,SVD,TD)')
parser.add_argument('-l', dest='library', action='store',type=str,help='Library (numpy,slepc)')
parser.add_argument('-p', dest='problem', action='store',type=str,help='Problem or fields')
parser.add_argument('-r', dest='nbasis', action='store',type=int,help='# of basis vectors')
parser.add_argument('-f', dest='filter', action='store',type=int,help='Basis filter')

args = parser.parse_args()

print('---- Reading data ----')

param = [1]
if (args.sets == "multi"):

    snap_files=[]
    snap_files = find_file("snp",snap_files)
    mesh_files=[]
    mesh_files = find_file("msh",mesh_files)

    param = []
    for iip in range(1,len(snap_files)+1):
        param.append(iip*0.001)
    data = reader(args.case,args.problem,len(param))
    data.readMesh2(mesh_files)
    data.readSnaps2(snap_files)
    data.setMean()
    data.center()

elif (args.sets == "single"):
    data = reader(args.case,args.problem,1)
    data.readMesh()
    data.readSnaps()
    data.setMean()
    data.center()

print('---- Creating reduced space ----')
basis = rSpace(args.method,args.typed,args.library,data.npar)
basis.setData(data.X,args.nbasis)
basis.create()
if args.filter ==1: basis.filter()
bPlot(basis)
print('		r = ',basis.r)

print('---- Projection error ----')
for iis in range(0,basis.r):
    basis.U[:,iis] = data.iLMass * basis.U[:,iis]

print('---- Writing basis ----')
bas_file = h5py.File(str(args.case)+'.rom.bas','w',driver='mpio',comm=MPI.COMM_WORLD)
#bas_file = h5py.File(str(args.case)+'.rom.bas','w')

grp = bas_file.create_group('Basis')
grp.attrs.create('R',basis.r)
dset = bas_file.create_dataset('Basis/EigenValues', data=basis.S[0:basis.r])

grpmean = bas_file.create_group('SnapshotMean')
grpmean.attrs.create('P',len(param))
dset = bas_file.create_dataset('SnapshotMean/Parameters', data=param[:])

iif = 0
for field in data.fields:
    #INS
    dsetMean = bas_file.create_dataset('SnapshotMean/'+str(field), data=data.XMean.T[:,iif:data.nn*data.nf:data.nf])
    dsetBasis = bas_file.create_dataset('Basis/'+str(field), data=basis.U.T[:,iif:data.nn*data.nf:data.nf])
    iif += 1

bas_file.close()

print("--- Total time: %s seconds ---" % (time.time() - start_time))
