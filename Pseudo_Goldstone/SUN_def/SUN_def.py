from sympy import sqrt, Symbol,symbols,conjugate,I,flatten,simplify,expand
from numpy import array,arange,triu,tril,nonzero,append,unique, vectorize,sum,prod


from ..util import MatrixProd,Deriv,Tuples


def GetAssumptions(Sym,assL):
    tmpA=[]
    for i in assL:
        try:
            tmpA.append(Sym.assumptions0[i] )
        except:
            tmpA.append(None )
    return tmpA

def Definitions(DimN, Gauge,Diag=False):
    global gauge, dimN
    dimN,gauge=DimN,Gauge
    global sqrt2,dimRange, mPhi2, mPhip2, v, vPhi, muH, lamH, lamHPhi, lamPhi
    mPhi2=array( symbols('mPhi2(1:{})(1:{})'.format(str(dimN+1),str(dimN+1)),complex=True,real=False      )         ).reshape(dimN,dimN)
    mPhi2[dimN-1][dimN-1]=Symbol('mPhi2{}{}'.format(dimN,dimN),real=True )#this is real, due to the minimization conditions
    mPhip2=array( symbols('mPhip2(1:{})(1:{})'.format(str(dimN+1),str(dimN+1)),complex=True,real=False      )         ).reshape(dimN,dimN)


    #make mPhi symmetric (faster than numpy.triu(mPhi,+1).T+numpy.triu(mPhi))
    for i in range(dimN):
        for j in range(i+1,dimN):
            mPhi2[j][i]=mPhi2[i][j]

    #make mPhip hermitian (faster than numpyconjugate(numpy.triu(mPhi,+1).T)+numpy.triu(mPhi))
    for i in range(dimN):
        for j in range(i+1,dimN):
            mPhip2[j][i]=conjugate(mPhip2[i][j])
    #make the diagonal  real. keep in mind that the squared elements of the diagonal are real.
    #So the elements can be either real or imaginary
    for i in range(dimN):
        exec( 'mPhip2[{}][{}]=Symbol(  \'mPhip2{}{}\'  ,real=True)'.format(str(i),str(i),str(i+1),str(i+1)) )



    tmpMPHI=(triu(mPhi2)).reshape(dimN**2)

    ParameterSymbols= array( [ (tmpMPHI[i], GetAssumptions(tmpMPHI[i],['complex','real','positive'] ) ) \
                                 for i in nonzero(tmpMPHI)[0]] )

    tmpMPHI=(triu(mPhip2)).reshape(dimN**2)
    ParameterSymbols=append(ParameterSymbols, array( [ (tmpMPHI[i], GetAssumptions(tmpMPHI[i],['complex','real','positive'] ) )\
                                                            for i in nonzero(tmpMPHI)[0]] ) )

    del tmpMPHI


    sqrt2=sqrt(2);

    if Diag:
        global Oh11,Oh12, MH,MS0,MG
        if gauge!='un':

            Fields=array( symbols('H S0 G1:{} G0 Gp Gm'.format(2*dimN) )   )
        else:
            Fields=array( symbols('H S0 G1:{}'.format(2*dimN) )   )


        MH,MS0 = symbols('MH MS0 ',real=True,positive=True)
        MG = symbols('MG1:{}'.format(2*dimN),real=True,positive=True)
        tmpMPHI=[MH,MS0]+list(MG)
        ParameterSymbols=append(ParameterSymbols, array( [ (tmpMPHI[i], GetAssumptions(tmpMPHI[i],['complex','real','positive'] ) ) \
                                     for i in nonzero(tmpMPHI)[0]] ))

        del tmpMPHI

        Oh11,Oh12=symbols('Oh11 Oh12',real=True)
        v=Symbol('v',positive=True);
        vPhi=Symbol('vPhi',positive=True);
        muH=Symbol('muH');
        lamH=Symbol('lamH',real=True,positive=True);
        lamHPhi=Symbol('lamHPhi',real=True,positive=None);
        lamPhi=Symbol('lamPhi',real=True,positive=True);


        ParameterSymbols=append(ParameterSymbols, array( [\
                                                                (Oh11,GetAssumptions(Oh11,['complex','real','positive'] )),\
                                                                (Oh12,GetAssumptions(Oh12,['complex','real','positive'] )),\
                                                                (v,GetAssumptions(v,['complex','real','positive'] )),\
                                                                (vPhi,GetAssumptions(vPhi,['complex','real','positive'] )),\
                                                                (lamH,GetAssumptions(lamH,['complex','real','positive'] )),\
                                                                (lamHPhi,GetAssumptions(lamHPhi,['complex','real','positive'] )),\
                                                                (lamPhi,GetAssumptions(lamPhi,['complex','real','positive'] ))]))

        return  Fields,ParameterSymbols
    else:
        global Gp, H0, Gm, H0t, h, G0, H, Ht, Phi, Phit, chi, rho, phi, s
        global  subsvev, subsexpand


        dimRange=arange(1,dimN+1);
        dimRangeM=range(dimN-1)




        Phi = symbols('Phi1:{}'.format(str(dimN+1)))
        Phit = symbols('Phi1:{}t'.format(str(dimN+1)))

        if gauge=='un':
            H0, H0t=symbols('H0, H0t')
            H = [0,H0];
            Ht = [0, H0t];

        else:
            H0,H0t,Gp,Gm,G0=symbols('H0,H0t,Gp,Gm,G0')
            H = [Gp,H0];
            Ht = [Gm, H0t];



        ##################--Declare symbols for expaned scalars
        phi = list(symbols('phi1:{}'.format(str(dimN))))
        s = list(symbols('s1:{}'.format(str(dimN))))
        h , chi, rho=symbols('h chi rho')




        v=Symbol('v',positive=True);
        vPhi=Symbol('vPhi',positive=True);
        muH=Symbol('muH');
        lamH=Symbol('lamH',real=True,positive=True);
        lamHPhi=Symbol('lamHPhi',real=True,positive=None);
        lamPhi=Symbol('lamPhi',real=True,positive=True);

        ParameterSymbols=append(ParameterSymbols, array( [\
                                                                (v,GetAssumptions(v,['complex','real','positive'] )),\
                                                                (vPhi,GetAssumptions(vPhi,['complex','real','positive'] )),\
                                                                (lamH,GetAssumptions(lamH,['complex','real','positive'] )),\
                                                                (lamHPhi,GetAssumptions(lamHPhi,['complex','real','positive'] )),\
                                                                (lamPhi,GetAssumptions(lamPhi,['complex','real','positive'] ))]))



        #Expand the fields at their vevs
        if gauge=='un':
            subsexpand =array(\
                         [(H0,(h+v)/sqrt2 ),(H0t,(h+v)/sqrt2 ),\
                          (Phi[dimN-1],(rho+ I*chi+vPhi)/sqrt2 ),\
                          (Phit[dimN-1],(rho-I*chi+vPhi)/sqrt2 )]+ \
                         [(Phi[i], (phi[i]+I*s[i])/sqrt2   ) for i in dimRangeM]+\
                         [(Phit[i],(phi[i]-I*s[i])/sqrt2)  for i in dimRangeM])
            Fields=array(flatten([h,rho,s,chi,phi]))


            subsvev = array(\
                          [(H0,v/sqrt2 ),(H0t,v/sqrt2 ),\
                          (Phi[dimN-1], vPhi/sqrt2 ),\
                          (Phit[dimN-1],vPhi/sqrt2 )]+ \
                         [(Phi[i], 0) for i in dimRangeM]+\
                         [(Phit[i],0) for i in dimRangeM])

        else:
            subsexpand = array(\
                          [(H0,(h+I*G0+v)/sqrt2 ),(H0t,(h-I*G0+v)/sqrt2 ),\
                          (Phi[dimN-1], (rho+I*chi+vPhi)/sqrt2 ),\
                          (Phit[dimN-1],(rho-I*chi+vPhi)/sqrt2 )]+ \
                         [(Phi[i], (phi[i]+I*s[i])/sqrt2) for i in dimRangeM]+\
                         [(Phit[i],(phi[i]-I*s[i])/sqrt2) for i in dimRangeM])

            Fields=array(flatten([h,rho,s,chi,phi,G0,Gp,Gm]))


            subsvev = array(\
                              [(H0,v/sqrt2 ),(H0t,v/sqrt2 ),\
                               (G0,0),(Gm,0),(Gp,0),\
                              (Phi[dimN-1], vPhi/sqrt2 ),\
                              (Phit[dimN-1],vPhi/sqrt2 )]+ \
                             [(Phi[i], 0) for i in dimRangeM]+\
                             [(Phit[i],0) for i in dimRangeM])


        return list(Fields),ParameterSymbols

def GetLagrangian(AllFields=False,Diag=False):
    if Diag==False:
        mPhi2C=[[conjugate(i) for i in x] for x in mPhi2]
        V0=-muH**2/2*MatrixProd([H,Ht])+lamH/2*MatrixProd([H,Ht])**2+lamPhi/2*MatrixProd([Phi,Phit])**2\
        +lamHPhi*MatrixProd([H,Ht])*MatrixProd([Phi,Phit] );

        Vsoft=MatrixProd([Phi,mPhi2,Phi])+MatrixProd([Phit,mPhi2C,Phit])+MatrixProd([Phit,mPhip2,Phi])


        V=(V0+Vsoft)#.subs(subsexpand)

        subsmin= [ (mPhi2[i][dimN-1], -mPhip2[dimN-1][i]/2 ) for i in range(0,dimN-1)]+ \
        [(muH, sqrt(v**2*lamH + vPhi**2*lamHPhi)),\
        (lamPhi,-(lamHPhi*v**2 + 2*mPhi2[dimN-1][dimN-1] + 2*mPhip2[dimN-1][dimN-1] + 2*conjugate(mPhi2[dimN-1][dimN-1]))/vPhi**2),\
        (conjugate(mPhi2[dimN-1][dimN-1]),mPhi2[dimN-1][dimN-1] )]


        constV=simplify((V.subs(subsmin).subs(subsvev)) )

        if AllFields!=False:
            try:
                CheckMinimizations(AllFields,V, constV, subsmin)
            except:
                print 'Something went wrong while checking the minimization. \nHave you passed the fields correctly? '

        LMassInt = -( (V.subs(subsmin)).subs(subsexpand) -constV );
        return LMassInt
    else:
        FHS=array(AllFields[:2])
        FG=array(AllFields[2:2*dimN+1])
        #h=H*Oh11 - Oh12*S0
        hH=Oh11*FHS[0]-Oh12*FHS[1]
        #-------------------------
        #rho=H*Oh12 + Oh11*S0
        RHO=Oh12*FHS[0]+Oh11*FHS[1]
        #--------------------------

        Phit_times_Phi= (MatrixProd([FG,FG]) + (vPhi+RHO)**2)/2
        if gauge=='un':
            HD=1/sqrt2*array([0,hH+v])
            HDt=HD
            Gsm=array([0,0,0])
        else:
            Gsm=array(AllFields[2*dimN+1:])
            HD=array([Gsm[1],1/sqrt2*(hH+v+I*Gsm[0])])
            HDt=array([Gsm[2],1/sqrt2*(hH+v-I*Gsm[0])])

        subsmin= [ (mPhi2[i][dimN-1], -mPhip2[dimN-1][i]/2 ) for i in range(0,dimN-1)]+ \
        [(muH, sqrt(v**2*lamH + vPhi**2*lamHPhi)),\
        (lamPhi,-(lamHPhi*v**2 + 2*mPhi2[dimN-1][dimN-1] + 2*mPhip2[dimN-1][dimN-1] + 2*conjugate(mPhi2[dimN-1][dimN-1]))/vPhi**2),\
        (conjugate(mPhi2[dimN-1][dimN-1]),mPhi2[dimN-1][dimN-1] )]

        L_int=-(lamH/2*(MatrixProd([HD,HDt])**2 )+lamPhi/2*(Phit_times_Phi**2 )\
        +lamHPhi*MatrixProd([HD,HDt])*Phit_times_Phi)

        #Remove linear interactions (it is minimized!)
        Subs0=[(i,0) for i in AllFields]

        L_const=-lamH*v**4/8 - lamHPhi*v**2*vPhi**2/4 - lamPhi*vPhi**4/8
        Linear_terms=( -Oh11*lamH*v**3/2 - Oh11*lamHPhi*v*vPhi**2/2 - Oh12*lamHPhi*v**2*vPhi/2 - Oh12*lamPhi*vPhi**3/2  )*FHS[0]\
                      +(-Oh11*lamHPhi*v**2*vPhi/2 - Oh11*lamPhi*vPhi**3/2 + Oh12*lamH*v**3/2 + Oh12*lamHPhi*v*vPhi**2/2)*FHS[1]


        #Remove 2-point interactions (the mass)

        P2_terms=(\
         (- Oh11**2*lamHPhi*v**2/2 - 3*Oh11**2*lamPhi*vPhi**2/2 + 2*Oh11*Oh12*lamHPhi*v*vPhi - 3*Oh12**2*lamH*v**2/2 - Oh12**2*lamHPhi*vPhi**2/2 )/2*FHS[1]**2\
         +( -Oh11**2*lamHPhi*v*vPhi + 3*Oh11*Oh12*lamH*v**2/2 - Oh11*Oh12*lamHPhi*v**2/2 + Oh11*Oh12*lamHPhi*vPhi**2/2 - 3*Oh11*Oh12*lamPhi*vPhi**2/2 + Oh12**2*lamHPhi*v*vPhi)*FHS[0]*FHS[1]\
        +(- 3*Oh11**2*lamH*v**2/2 - Oh11**2*lamHPhi*vPhi**2/2 - 2*Oh11*Oh12*lamHPhi*v*vPhi - Oh12**2*lamHPhi*v**2/2 - 3*Oh12**2*lamPhi*vPhi**2/2)/2*FHS[0]**2
        -(lamH*v**2 +lamHPhi*vPhi**2)/4*Gsm[0]**2\
        -( lamH*v**2 + lamHPhi*vPhi**2)/2*Gsm[2]*Gsm[1]\
        )+\
        ( MatrixProd([FG,FG]))*( - lamHPhi*v**2/2 - lamPhi*vPhi**2/2 )/2

        #Include the mases (it is the diagonalized Lagrangian)

        L_mass=-(FHS[0]**2*MH**2 + FHS[1]**2*MS0**2)/2 -sum([FG[i-1]**2*MG[i-1]**2 for i in range(1,2*dimN) ])/2


        return expand(L_int-Linear_terms-P2_terms- L_const).subs(subsmin)+L_mass


    global tmp2
    def tmp2(i):
        if i[0]==i[1]:
            f=1/2.
        else:
            f=1
        return f*prod(i)*Deriv(L,i).subs(Subs0)

def CheckMinimizations(AllFields,V, constV, subsmin):#uses only global
    subs0=[ (i,0) for i in AllFields]

    print 'Checking vanishing of the first derivatives of the potential...'

    minV=unique(map(lambda i: \
             simplify(Deriv(V.subs(subsexpand),i ).subs(subs0).subs(subsmin) ),AllFields))
    if (minV==0).all():
        print 'The conditions are correct!'
    else:
        print 'The potential is not minimized correctlly...'
