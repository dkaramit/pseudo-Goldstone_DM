from sympy import sqrt, Symbol,symbols,conjugate,I,flatten,simplify,expand
from numpy import array,arange,triu,tril,nonzero,append,unique,prod,sum
import os.path



def prtcls(prts):
    return  tuple(sorted( prts ) )

def FRules(N_Point):
    N=len(N_Point[0][0])
    NPoint_dict={}
    if N==2:
        for i in N_Point:
            NPoint_dict[prtcls( map( str, i[0] ) ) ]=i[1]*(-i[2])
    else:
        for i in N_Point:
            NPoint_dict[prtcls( map( str, i[0] ) ) ]=i[1]*(-I*i[2])


    return NPoint_dict

def Make_Feynman_Rules(NPoint_dict):
    global DictP
    DictP={}
    for k in NPoint_dict.keys():

        DictP[k] = FRules(NPoint_dict[k])

def VertexValue(*particles):
    lp=len(particles)

    try:
        return DictP[lp][ prtcls(   map(str, particles)  ) ]
        #return eval('DictP'+str(lp)+'[ prtcls(   map(str, particles)  ) ]'   )
    except:
        return 0

def CheckInteractions(N_Point_dict, Initial_Lagrangian,AllFields):

    if  Initial_Lagrangian!=False and list(AllFields)!=False:
        testL=True
    else:
        testL=False

    if testL:
        global    LMassIntfinal, L_in
        print 'Checking Vertices...'
        LMassIntfinal=0
        SUBS0=[ (i,0) for i in AllFields]

        def tmp_p(k):
            return prod(k[0])*k[1]

        LMassIntfinal =expand(sum( [ sum(map(tmp_p, N_Point_dict[i])) for i in N_Point_dict.keys() ] ))

        L_in=expand(Initial_Lagrangian-simplify(Initial_Lagrangian.subs(SUBS0)))


        if (LMassIntfinal-L_in)==0:
            print 'The interactions have been identified correctly!!'
        else:
            # if (LMassIntfinal-L_in)!=0, try simplifying it and chech again!
            if (simplify(LMassIntfinal-L_in))==0:
                print 'The interactions have been identified correctly!!'
            else:
                print 'The final Lagrangian is not the same as the initial one... (check it!)'

def StoreVert(N_Points,AllFields,AllParameterSymbols,dimN,gauge,Directory='Frules'):

    print 'Writing Vertices (Feynman Rules and mass matrix entries)...'
    dirV=Directory

    if not os.path.exists(dirV):
        os.makedirs(dirV)

    if not os.path.exists(dirV+"/SU" + str(dimN)):
        os.makedirs(dirV+"/SU" + str(dimN))




    files=N_Points.keys()

    tmp =open(dirV+"/SU" + str(dimN)+ "/SU" + str(dimN) +'_'+gauge+ ".fields","w")
    [tmp.write(str(ff)+'\n') for ff in AllFields]

    tmp =open(dirV+"/SU" + str(dimN)+ "/SU" + str(dimN)+'_'+gauge+".parameters","w")
    [tmp.write(str(ff)+'\n') for ff in AllParameterSymbols]

    for file in files:
        tmp = open(dirV+"/SU" + str(dimN)+ "/SU" + str(dimN)+"_" +str(file)+"-point_"+gauge + ".vrt","w")
        if file==2:
            factorI=-1
        else:
            factorI=-I
        for i in N_Points[file]:
            particles=str(i[0])

            vertex=str(factorI*i[1]*i[2])

            line='{:<40} {:<40} {:<0}'.format(particles, '|' , vertex)
            #tmp.write( particles +"|\t|"+ vertex + "\n" )
            tmp.write( line +'\n')
        tmp.close()
    print 'All Done!'
