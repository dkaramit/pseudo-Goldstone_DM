from sympy import factorial , simplify,expand
from numpy import product , array ,  nonzero

import numpy as np
from multiprocessing import Pool


from ..util import Deriv , Tuples

def DEF_TMP(Langrangian,Fields):
    set_fields_to_0=[(i,0) for i in Fields  ]
    global TMP_int
    def TMP_int(particles):

        SymF=product([ factorial(particles.count(j)) for j in set(particles)])
        tmpval=1/SymF*simplify(Deriv(Langrangian,particles).subs(set_fields_to_0))
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
            FR=array(p.map(TMP_int,tmpTuples))
            Point_N[i]= [FR[TMPI] for TMPI  in nonzero(FR)[0] ]
            p.close()
            del p,FR
        else:
            FR=array(map(TMP_int,tmpTuples))
            Point_N[i]= [FR[TMPI] for TMPI  in nonzero(FR)[0] ]
            del FR
    return Point_N
