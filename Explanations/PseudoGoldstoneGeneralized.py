import sympy as sp
import numpy as np
from itertools import combinations_with_replacement as itTuples
import os.path
from multiprocessing import Pool

def Tuples(List,k):
    return list(itTuples(List,k))



def MatrixProd(a):
    n=len(a)-1
    if n!=0:
        return np.dot(MatrixProd(a[:n]),a[n])
    else:
        return a[0]


def Deriv(L,a):
    try:
        
        n=len(a)-1
        if n>=0:
            return sp.diff(Deriv(L,a[:n]),a[n])
        else:
            return L
    except:
        return sp.diff(L,a)
    
    
def GetAssumptions(Sym,assL):
    tmpA=[]
    for i in assL:
        try:
            tmpA.append(Sym.assumptions0[i] )
        except:
            tmpA.append(None )
    return tmpA



def Definitions(DimN, Gauge):
    global gauge, dimN
    global dimRange, indexRange, mPhi2, mPhip2, v, vPhi, muH, lamH, lamHPhi, lamPhi
    global Gp, H0, Gm, H0t, h, G0, H, Ht, Phi, Phit, chi, rho, phi, s
    global  sqrt2, subsvev, subsexpand
    
    '''gauge, dimN, dimRange, indexRange, mPhi2, mPhip2, v, vPhi, muH, lamH, lamHPhi, lamPhi,\
            Gp, H0, Gm, H0t, h, G0, H, Ht, Phi, Phit, chi, rho, phi, s,\
            sqrt2, subsvev, subsexpand=0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
            0, 0, 0'''

    

    
    dimN=DimN
    gauge=Gauge
    
    dimRange=np.arange(1,dimN+1);
    dimRangeM=range(dimN-1)
    indexRange=range(0,dimN);

    sqrt2=sp.sqrt(2);


    
    
    mPhi2=np.array( sp.symbols('mPhi2(1:{})(1:{})'.format(str(dimN+1),str(dimN+1)),complex=True,real=False      )         ).reshape(dimN,dimN)
    mPhi2[dimN-1][dimN-1]=sp.Symbol('mPhi2{}{}'.format(dimN,dimN),real=True )#this is real, due to the minimization conditions
    mPhip2=np.array( sp.symbols('mPhip2(1:{})(1:{})'.format(str(dimN+1),str(dimN+1)),complex=True,real=False      )         ).reshape(dimN,dimN)
    

    #make mPhi symmetric (faster than np.triu(mPhi,+1).T+np.triu(mPhi))
    for i in range(dimN):
        for j in range(i+1,dimN):
            mPhi2[j][i]=mPhi2[i][j]
    
    #make mPhip hermitian (faster than np.conjugate(np.triu(mPhi,+1).T)+np.triu(mPhi))
    for i in range(dimN):
        for j in range(i+1,dimN):
            mPhip2[j][i]=sp.conjugate(mPhip2[i][j])
    #make the diagonal  real. keep in mind that the squared elements of the diagonal are real. 
    #So the elements can be either real or imaginary          
    for i in range(dimN):
        exec( 'mPhip2[{}][{}]=sp.Symbol(  \'mPhip2{}{}\'  ,real=True)'.format(str(i),str(i),str(i+1),str(i+1)) )

       
    tmpMPHI=(np.triu(mPhi2)).reshape(dimN**2)
    
    ParameterSymbols= np.array( [ (tmpMPHI[i], GetAssumptions(tmpMPHI[i],['complex','real','positive'] ) ) \
                                 for i in np.nonzero(tmpMPHI)[0]] ) 

    tmpMPHI=(np.triu(mPhip2)).reshape(dimN**2)
    ParameterSymbols=np.append(ParameterSymbols, np.array( [ (tmpMPHI[i], GetAssumptions(tmpMPHI[i],['complex','real','positive'] ) )\
                                                            for i in np.nonzero(tmpMPHI)[0]] ) )
    del tmpMPHI
    #print EverySymbol
    
    Phi = sp.symbols('Phi1:{}'.format(str(dimN+1)))
    Phit = sp.symbols('Phi1:{}t'.format(str(dimN+1)))

    if gauge=='un':
        H0, H0t=sp.symbols('H0, H0t')
        H = [0,H0];
        Ht = [0, H0t];
        
    else:    
        H0,H0t,Gp,Gm,G0=sp.symbols('H0,H0t,Gp,Gm,G0')
        H = [Gp,H0];
        Ht = [Gm, H0t];

    
    
    ##################--Declare symbols for expaned scalars 
    phi = list(sp.symbols('phi1:{}'.format(str(dimN))))
    s = list(sp.symbols('s1:{}'.format(str(dimN))))
    h , chi, rho=sp.symbols('h chi rho')
    
    
    
    
    v=sp.Symbol('v',positive=True);
    vPhi=sp.Symbol('vPhi',positive=True);
    muH=sp.Symbol('muH');
    lamH=sp.Symbol('lamH',real=True,positive=True);
    lamHPhi=sp.Symbol('lamHPhi',real=True,positive=None);
    lamPhi=sp.Symbol('lamPhi',real=True,positive=True);

    ParameterSymbols=np.append(ParameterSymbols, np.array( [\
                                                            (v,GetAssumptions(v,['complex','real','positive'] )),\
                                                            (vPhi,GetAssumptions(vPhi,['complex','real','positive'] )),\
                                                            (lamH,GetAssumptions(lamH,['complex','real','positive'] )),\
                                                            (lamHPhi,GetAssumptions(lamHPhi,['complex','real','positive'] )),\
                                                            (lamPhi,GetAssumptions(lamPhi,['complex','real','positive'] ))]))

   
    
    #Expand the fields at their vevs
    if gauge=='un':
        subsexpand =np.array(\
                     [(H0,(h+v)/sqrt2 ),(H0t,(h+v)/sqrt2 ),\
                      (Phi[dimN-1],(rho+ sp.I*chi+vPhi)/sqrt2 ),\
                      (Phit[dimN-1],(rho-sp.I*chi+vPhi)/sqrt2 )]+ \
                     [(Phi[i], (phi[i]+sp.I*s[i])/sqrt2   ) for i in dimRangeM]+\
                     [(Phit[i],(phi[i]-sp.I*s[i])/sqrt2)  for i in dimRangeM])
                
        Fields=np.array(sp.flatten([h,rho,s,chi,phi]))        
        subsvev = np.array(\
                      [(H0,v/sqrt2 ),(H0t,v/sqrt2 ),\
                      (Phi[dimN-1], vPhi/sqrt2 ),\
                      (Phit[dimN-1],vPhi/sqrt2 )]+ \
                     [(Phi[i], 0) for i in dimRangeM]+\
                     [(Phit[i],0) for i in dimRangeM])    
    
    else:
        subsexpand = np.array(\
                      [(H0,(h+sp.I*G0+v)/sqrt2 ),(H0t,(h-sp.I*G0+v)/sqrt2 ),\
                      (Phi[dimN-1], (rho+sp.I*chi+vPhi)/sqrt2 ),\
                      (Phit[dimN-1],(rho-sp.I*chi+vPhi)/sqrt2 )]+ \
                     [(Phi[i], (phi[i]+sp.I*s[i])/sqrt2) for i in dimRangeM]+\
                     [(Phit[i],(phi[i]-sp.I*s[i])/sqrt2) for i in dimRangeM])
        
        Fields=np.array(sp.flatten([h,rho,s,chi,phi,G0,Gp,Gm]))
        
        
        subsvev = np.array(\
                          [(H0,v/sqrt2 ),(H0t,v/sqrt2 ),\
                           (G0,0),(Gm,0),(Gp,0),\
                          (Phi[dimN-1], vPhi/sqrt2 ),\
                          (Phit[dimN-1],vPhi/sqrt2 )]+ \
                         [(Phi[i], 0) for i in dimRangeM]+\
                         [(Phit[i],0) for i in dimRangeM])
        
    
    return list(Fields),ParameterSymbols

def GetLagrangian(AllFields=False):
    #global V, constV, subsmin#these are for internal checks. Not really useful
    
    mPhi2C=[[sp.conjugate(i) for i in x] for x in mPhi2]

    V0=-muH**2/2*MatrixProd([H,Ht])+lamH/2*MatrixProd([H,Ht])**2+lamPhi/2*MatrixProd([Phi,Phit])**2\
    +lamHPhi*MatrixProd([H,Ht])*MatrixProd([Phi,Phit] );

    Vsoft=MatrixProd([Phi,mPhi2,Phi])+MatrixProd([Phit,mPhi2C,Phit])+MatrixProd([Phit,mPhip2,Phi])


    V=(V0+Vsoft)#.subs(subsexpand)
           
    subsmin= [ (mPhi2[i][dimN-1], -mPhip2[dimN-1][i]/2 ) for i in range(0,dimN-1)]+ \
    [(muH, sp.sqrt(v**2*lamH + vPhi**2*lamHPhi)),\
    (lamPhi,-(lamHPhi*v**2 + 2*mPhi2[dimN-1][dimN-1] + 2*mPhip2[dimN-1][dimN-1] + 2*sp.conjugate(mPhi2[dimN-1][dimN-1]))/vPhi**2),\
    (sp.conjugate(mPhi2[dimN-1][dimN-1]),mPhi2[dimN-1][dimN-1] )]

    
    constV=sp.simplify((V.subs(subsmin).subs(subsvev)) )
    
    if AllFields!=False:
        try:
            CheckMinimizations(AllFields,V, constV, subsmin)
        except:
            print 'Something went wrong while checking the minimization. \nHave you passed the fields correctly? '
    
    LMassInt = -( (V.subs(subsmin)).subs(subsexpand) -constV );
    return LMassInt



def CheckMinimizations(AllFields,V, constV, subsmin):#uses only global
    subs0=[ (i,0) for i in AllFields]
    
    print 'Checking vanishing of the first derivatives of the potential...'
    
    minV=np.unique(map(lambda i: \
             sp.simplify(Deriv(V.subs(subsexpand),i ).subs(subs0).subs(subsmin) ),AllFields))
    if (minV==0).all():
        print 'The conditions are correct!'
    else:
        print 'The potential is not minimized correctlly...'

#Pool needs defined functions at the top level. So we need to define a functions which defines TMP_int (called in 
#IdentifyInteractions)

def DEF_TMP(Langrangian,Fields):
    set_fields_to_0=[(i,0) for i in Fields  ]
    global TMP_int
    def TMP_int(particles):

        SymF=np.product([ sp.factorial(particles.count(j)) for j in set(particles)])
        tmpval=1/SymF*sp.simplify(Deriv(Langrangian,particles).subs(set_fields_to_0))
        if tmpval!=0:
            return [particles, tmpval,SymF] 
        else:
            return 0

OPTIONS_Int=['Parallel']
DEF_OPT_Int={'Parallel':True}
def IdentifyInteractions(Langrangian,All_Fields,**opts):

    
    
    #----------------Begin check opts    
    if len(opts) == 0:
        print 'Using default options...'
        opts=DEF_OPT_Int
        

            
    for i in opts:
        if not (i in OPTIONS_Int):
            print 'invalid option '+i
            print 'availabe options: ' 
            print OPTIONS_Int
            return 'ERR:: invalid option. Abort!'
        
    xtmp=opts.copy() 
    for i in OPTIONS_Int:
        if not (i in opts):
            xtmp.update({i:DEF_OPT_Int[i]})
    
    Parallel=xtmp['Parallel']
    
    if Parallel!=True:
        Parallel=False

        
    #----------------End check opts    


    #extract all interactions  involving from Min_in to Max_int particles
    Min_int=2
    Max_int=4



    Point_N={}
    DEF_TMP(Langrangian,All_Fields)
        
    ###########################################################
    for i in range(Min_int,Max_int+1):   
        tmpTuples=Tuples(All_Fields,i)
        print 'calculating {}-point interactions'.format(i)
        
        if Parallel:
            p=Pool()
            FR=np.array(p.map(TMP_int,tmpTuples))
            Point_N[i]= [FR[TMPI] for TMPI  in np.nonzero(FR)[0] ]
            p.close()
            del p,FR
        else:
            FR=np.array(map(TMP_int,tmpTuples))
            Point_N[i]= [FR[TMPI] for TMPI  in np.nonzero(FR)[0] ]
            del FR
    return Point_N

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
            NPoint_dict[prtcls( map( str, i[0] ) ) ]=i[1]*(-sp.I*i[2])
        
        
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
    
    if N_Point_dict!=False and Initial_Lagrangian!=False and AllFields!=False:
        testL=True
    else:
        testL=False

    if testL:
        global    LMassIntfinal, L_in
        print 'Checking Vertices...'
        LMassIntfinal=0
        SUBS0=[ (i,0) for i in AllFields]

        for TypeOfVert in N_Point_dict.keys():
            TypeV=N_Point_dict[TypeOfVert]
            
            LMassIntfinal+=np.sum([ np.product(tmpi[0])*tmpi[1] for tmpi in TypeV])
        L_in=Initial_Lagrangian-sp.simplify(Initial_Lagrangian.subs(SUBS0)) 
        
                
        if (sp.simplify(LMassIntfinal-L_in))==0:
            print 'The interactions have been identified correctly!!'
        else:
            print 'The final Lagrangian is not the same as the initial one... (check it!)'    
    #return LMassIntfinal


    
    










def StoreVert(N_Points,AllFields,AllParameterSymbols,Directory='Frules'):
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
            factorI=-sp.I
        for i in N_Points[file]:
            particles=str(i[0])

            vertex=str(factorI*i[1]*i[2])

            line='{:<40} {:<40} {:<0}'.format(particles, '|' , vertex)
            #tmp.write( particles +"|\t|"+ vertex + "\n" ) 
            tmp.write( line +'\n')
        tmp.close()        
    print 'All Done!'
