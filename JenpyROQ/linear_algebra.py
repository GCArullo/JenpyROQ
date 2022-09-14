import numpy as np
    
## General linear algebra routines

def scalar_product(vec1, vec2, df, weights=1.):

    vec1 = vec1 * 1./np.sqrt(weights)
    vec2 = vec2 * 1./np.sqrt(weights)

    return 4.*df*np.real(np.vdot(vec1,vec2))

def normalise_vector(vec, df):

    norm = np.sqrt(scalar_product(vec, vec, df))

    if not(norm==0.0): return vec/norm
    else             : return vec

def projection(u, v):
    
    return u * np.vdot(v,u)

def gram_schmidt(basis, vec, df):
    
    """
        Calculating the normalized residual (= a new basis term) of a vector `vec` from the known `basis`.
    """
    
    for i in np.arange(0,len(basis)):
        vec = vec - projection(basis[i], vec)
    
    return normalise_vector(vec, df)
