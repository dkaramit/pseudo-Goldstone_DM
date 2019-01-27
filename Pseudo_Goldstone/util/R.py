import sympy as sp
import numpy as np
from MatrixProd import MatrixProd
from multiprocessing import Pool

#R_keywords=['Parametrization']
R_Deafault_Options={'Parametrization':'tan'}
R_Available_Options={'Parametrization':['tan','cs','Os']}

def R(N,i,j,**opts):
    tmp_opts={}
    for op_k in opts.keys():
        if op_k in R_Deafault_Options.keys():
            if opts[op_k] in R_Available_Options[op_k]:
                tmp_opts[op_k]=opts[op_k]
            else:
                print 'This option for {} is not available.\nAvailable options: {}'.format(op_k,\
                                                                                R_Available_Options)
                return 'Abort!'
        else:
            print 'Wrong keyword {0}. \nAvailable keywords: {1}.\nWith\
        {2}'.format(op_k,R_Deafault_Options.keys(),R_Available_Options)
            return 'Abort!'

    for d_op_k in R_Deafault_Options.keys():
        if not (d_op_k in opts.keys()):
            tmp_opts[d_op_k]=R_Deafault_Options[d_op_k]


    if i == j or i>N or i==0 or j==0:
        print 'Plane indices incorrect.'
        return 'Error'


    else:
        if i>j:
            i1=j
            i2=i
        if i<j:
            i1=i
            i2=j
        if tmp_opts['Parametrization']=='tan':
            t=sp.Symbol('t{0}{1}'.format(i1,i2),real=True)
            c=1/sp.sqrt(t**2+1)
            s=t/sp.sqrt(t**2+1)

        if tmp_opts['Parametrization']=='cs':
            t=sp.Symbol('theta_{0}{1}'.format(i1,i2),real=True)
            c,s=sp.cos(t),sp.sin(t)

        if tmp_opts['Parametrization']=='Os':
            O11=sp.Symbol('O_{0}{1}'.format(i1,i2),real=True)
            c,s=O11,sp.sqrt(1-O11**2)




        tmpM=np.identity(N,dtype=object)


        tmpM[i1-1][i1-1]=c
        tmpM[i2-1][i2-1]=c
        tmpM[i1-1][i2-1]=-s
        tmpM[i2-1][i1-1]=s
        return np.matrix(tmpM)




def R_call(N,param):
    global RR
    def RR(ind):
        return R(N,ind[0],ind[1],Parametrization=param)


def OrthogonalMatrix(N,param='tan'):

    R_call(N,param)
    p=Pool()
    inds=[ (i,j)  for i in range(1,N+1) for j in range(i+1,N+1)]
    Rs=p.map(RR,inds)
    p.close()

    try:
        return MatrixProd(Rs)
    except:
        print 'Something went wrong!'
        return 'Abort!'
