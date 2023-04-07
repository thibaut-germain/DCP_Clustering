import numpy as np
cimport numpy as np

from libc.float cimport DBL_MAX


cdef inline double square_euclidean(np.float64_t x, np.float64_t y): return (x-y)**2


cimport cython

def cost_matrix(np.ndarray[np.float64_t,ndim=2]x, np.ndarray[np.float64_t,ndim=2]y, int w=-1):

    cdef int lx,ly,i,j,start,stop,width
    lx = x.shape[0]
    ly = y.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2] cm = np.zeros((lx,ly),dtype=np.float64) + DBL_MAX

    #initialisation
    cm[0,0] = square_euclidean(x[0,0],y[0,0])

    if w < 0: 
        i=0
        j=0
        for i in range(1,lx): 
            cm[i,0] = cm[i-1,0]+square_euclidean(x[i,0],y[0,0])
        for j in range(1,ly): 
            cm[0,j] = cm[0,j-1]+square_euclidean(x[0,0],y[j,0])
        i=0
        j=0
        for i in range(1,lx): 
            for j in range(1,ly):
                cm[i,j] = square_euclidean(x[i,0],y[j,0]) + min(cm[i-1,j-1],cm[i,j-1],cm[i-1,j])

    elif w>0: 
        i=0
        j=0
        for i in range(1,min(w,lx)): 
            cm[i,0] = cm[i-1,0]+square_euclidean(x[i,0],y[0,0])
        for j in range(1,min(w,ly)):
            cm[0,j] = cm[0,j-1]+square_euclidean(x[0,0],y[j,0]) 

        if lx>ly: 
            i=0
            j=0
            width = lx-ly+w
            for j in range(1,ly):
                start = max(1,j-w)
                stop = min(lx,j+width)
                for i in range(start,stop):
                    cm[i,j] = square_euclidean(x[i,0],y[j,0]) + min(cm[i-1,j-1],cm[i,j-1],cm[i-1,j])
        
        else: 
            i=0
            j=0
            width = ly-lx+w
            for i in range(1,lx):
                start = max(1,i-w)
                stop = min(ly,i+width)
                for j in range(start,stop): 
                    cm[i,j] = square_euclidean(x[i,0],y[j,0]) + min(cm[i-1,j-1],cm[i,j-1],cm[i-1,j])
    else: 
        raise ValueError('Sakoe-Chiba constraint must be greater or egual to one.')
    return cm


def warping_path(np.ndarray[np.float64_t,ndim=2]cm): 
    cdef np.int64_t i,j
    cdef int n_step
    cdef np.float64_t min_val


    n_step = 0
    i = cm.shape[0]-1
    j = cm.shape[1]-1

    cdef np.ndarray[np.int64_t,ndim=2] path = np.zeros((i+j+1,2),dtype = np.int64)
    path[0,0] = i
    path[0,1] = j

    while (i!=0) or (j!=0): 
        n_step +=1
        if i == 0 and j > 0 :
            j -=1 
        elif j == 0 and i > 0 : 
            i-=1
        else: 
            min_val = min(cm[i-1,j-1],cm[i,j-1],cm[i-1,j])
            if cm[i-1,j-1] == min_val:
                i-=1
                j-=1
            elif cm[i,j-1] == min_val:
                j-=1
            else: 
                i-=1
        path[n_step,0] =i
        path[n_step,1] =j

    return path[:n_step+1,:][::-1]

      
def single_valence_vector(np.ndarray[np.int64_t,ndim=2]path,barycenter_size): 
    #be carefull path first column correspond to the barycenter
    cdef np.ndarray[np.float64_t,ndim=2] vm = np.zeros((barycenter_size,1),dtype=np.float64)
    cdef int i,n_step

    n_step = path.shape[0]
    for i in range(n_step): 
        vm[path[i,0],0]+=1     

    return vm


def single_warping_vector(np.ndarray[np.int64_t,ndim=2]path, np.ndarray[np.float64_t,ndim=2]X_i, barycenter_size): 

    cdef np.ndarray[np.float64_t,ndim=2] sw = np.zeros((barycenter_size,1),dtype=np.float64)
    cdef int n_step,i 

    n_step = path.shape[0]

    for i in range(n_step): 
        sw[path[i,0],0] += X_i[path[i,1],0]

    return sw