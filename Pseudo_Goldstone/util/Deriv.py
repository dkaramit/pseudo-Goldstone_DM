from sympy import diff


def Deriv(L,a):
    try:

        n=len(a)-1
        if n>=0:
            return diff(Deriv(L,a[:n]),a[n])
        else:
            return L
    except:
        return diff(L,a)
