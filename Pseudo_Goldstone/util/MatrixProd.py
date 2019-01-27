from  numpy import dot

def MatrixProd(a):
    n=len(a)-1
    if n!=0:
        return dot(MatrixProd(a[:n]),a[n])
    else:
        return a[0]
