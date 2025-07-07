import numpy as np



def DMD(data,r):
    
    X1 = data[:, :-1] 
    X2 = data[:, 1:]
    
    N,T = data.shape

    U,S,V = np.linalg.svd(X1,full_matrices=0) # Step 1
    Ur = U[:,:r] 
    Sr = np.diag(S[:r])
    Vr = V[:r,:]
    Atilde = np.linalg.solve(Sr.T,(Ur.T @ X2 @ Vr.T).T).T # Step 2 #check
    D, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(D)
    


    alpha1 = Sr @ Vr[:,0]
    Phi = X2 @ np.linalg.solve(Sr.T,Vr).T @ W # Step 4
    

    x1 = data[:,0]
    x1 = x1.reshape(-1,1)

    #b = np.linalg.solve(W @ Lambda,alpha1)
    A = Phi @ Lambda @ np.linalg.pinv(Phi)
    
    return Phi, A

def DMDStatePrediction(data, r, pred_step):
    _, _, A, = DMD(data, r)
    N,T = data.shape
    mat = np.append(data, np.zeros((N, pred_step)), axis = 1)
    for t in range(pred_step):
        mat[:, T + t] = (A @ mat[:, T + t - 1]).real
        
    return mat[:,-pred_step:]


def DMD_Recon(data,r, t):
    dt = t[2] - t[1]

    X1 = data[:, :-1] 
    X2 = data[:, 1:]

    N,T = data.shape

    U,S,V = np.linalg.svd(X1,full_matrices=0) # Step 1
    Ur = U[:,:r] 
    Sr = np.diag(S[:r])
    Vr = V[:r,:]
    Atilde = np.linalg.solve(Sr.T,(Ur.T @ X2 @ Vr.T).T).T # Step 2 #check
    D, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(D)
    
    omega = np.emath.log(Lambda)/dt
    omega[~np.isfinite(omega)] = 0 
    


    alpha1 = Sr @ Vr[:,0]
    Phi = X2 @ np.linalg.solve(Sr.T,Vr).T @ W # Step 4
    

    x1 = data[:,0]
    x1 = x1.reshape(-1,1)

    b = np.linalg.solve(W @ Lambda,alpha1)

    s = (r, len(t))



    Psi = np.zeros([r, len(t)], dtype='complex')
    for i,_t in enumerate(t):
        Psi[:,i] = np.multiply(np.power(D, _t/dt), b)

    X_dmd = Phi@Psi 
    
    
    return Lambda, omega, Psi, Phi

def ExactDMD(D, r, t): 
    # create DMD input-output matrices
    X = D[:,:-1]
    Y = D[:,1:]

    # SVD of input matrix
    U2,Sig2,Vh2 = np.linalg.svd(X, False)

    U = U2[:,:r]
    Sig = np.diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]

    # build A tilde
    Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.inv(Sig))
    mu,W = np.linalg.eig(Atil)
    
    Lambda = np.diag(mu)
    dt = t[2] - t[1]
    omega = np.emath.log(Lambda)/dt
    omega[~np.isfinite(omega)] = 0 

    # build DMD modes
    Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W)

    # compute time evolution
    b = np.dot(np.linalg.pinv(Phi), X[:,0])
    Psi = np.zeros([r, len(t)], dtype='complex')
    for i,_t in enumerate(t):
        Psi[:,i] = np.multiply(np.power(mu, _t/dt), b)
    
    return Phi, Psi, omega
    
def fbExactDMD(D, r, t): 
    # create DMD input-output matrices
    X = D[:,:-1]
    Y = D[:,1:]

    # SVD of X matrix
    U2,Sig2,Vh2 = np.linalg.svd(X, False)

    U = U2[:,:r]
    Sig = np.diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]

    # build A tilde
    f_Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.inv(Sig))
    
    # SVD of Y matrix
    U2,Sig2,Vh2 = np.linalg.svd(X, False)

    U = U2[:,:r]
    Sig = np.diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    
    b_Atil = np.dot(np.dot(np.dot(U.conj().T, X), V), np.linalg.inv(Sig))
 
    Atil =  np.emath.sqrt(np.multiply(f_Atil, np.linalg.inv(b_Atil)))
    mu,W = np.linalg.eig(Atil)
    
    Lambda = np.diag(mu)
    dt = t[2] - t[1]
    omega = np.emath.log(Lambda)/dt
    omega[~np.isfinite(omega)] = 0 

    # build DMD modes
    Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W)

    # compute time evolution
    b = np.dot(np.linalg.pinv(Phi), X[:,0])
    Psi = np.zeros([r, len(t)], dtype='complex')
    for i,_t in enumerate(t):
        Psi[:,i] = np.multiply(np.power(mu, _t/dt), b)
    
    return Phi, Psi, omega
    


def createHankel(data, num_columns):
    
    X = np.lib.stride_tricks.sliding_window_view(data, num_columns)
        
    return X

def createHankel2(data, num_columns):
    
    X = np.array([data[i:i+num_columns] for i in range(len(data)-num_columns+1)])
        
    return X

def fbDMD(data,r, t):
    dt = t[2] - t[1]

    X1 = data[:, :-1] 
    X2 = data[:, 1:]

    N,T = data.shape

    U,S,V = np.linalg.svd(X1,full_matrices=0) # Step 1
    Ur = U[:,:r] 
    Sr = np.diag(S[:r])
    Vr = V[:r,:]
    
    f_Atilde = np.linalg.solve(Sr.T,(Ur.T @ X2 @ Vr.T).T).T
    
    
    U,S,V = np.linalg.svd(X2,full_matrices=0) 
    b_Atilde = np.linalg.solve(Sr.T,(Ur.T @ X1 @ Vr.T).T).T
    
    
    Atilde = np.emath.sqrt(np.multiply(f_Atilde, np.linalg.inv(b_Atilde)))
    #Atilde = np.exp(0.5*(np.log(f_Atilde) + np.log(np.linalg.inv(b_Atilde))))
    
    D, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(D)
    
    omega = np.emath.log(Lambda)/dt
    omega[~np.isfinite(omega)] = 0 
    


    alpha1 = Sr @ Vr[:,0]
    Phi = X2 @ np.linalg.solve(Sr.T,Vr).T @ W # Step 4
    

    x1 = data[:,0]
    x1 = x1.reshape(-1,1)

    b = np.linalg.solve(W @ Lambda,alpha1)

    s = (r, len(t))



    Psi = np.zeros([r, len(t)], dtype='complex')
    for i,_t in enumerate(t):
        Psi[:,i] = np.multiply(np.power(D, _t/dt), b)

    X_dmd = Phi@Psi 
    
    
    return Lambda, omega, Psi

def tlsDMD(data,r, t):
    dt = t[2] - t[1]

    X1 = data[:, :-1] 
    X2 = data[:, 1:]
    
    Z = np.concatenate((X1, X2), axis = 0)

    N,T = data.shape

    U,S,V = np.linalg.svd(Z,full_matrices=0) # Step 1
    U11 = U[:r,:r] 
    U21 = U[:r+1, :r]
    Sr = np.diag(S[:r])
    Vr = V[:r,:]
    
    
    Atilde = U21 * np.linalg.pinv(U11)
  
    
    D, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(D)
    
    omega = np.emath.log(Lambda)/dt
    omega[~np.isfinite(omega)] = 0 
    


    alpha1 = Sr @ Vr[:,0]
    Phi = X2 @ np.linalg.solve(Sr.T,Vr).T @ W # Step 4
    

    x1 = data[:,0]
    x1 = x1.reshape(-1,1)

    b = np.linalg.solve(W @ Lambda,alpha1)

    s = (r, len(t))



    Psi = np.zeros([r, len(t)], dtype='complex')
    for i,_t in enumerate(t):
        Psi[:,i] = np.multiply(np.power(D, _t/dt), b)

    X_dmd = Phi@Psi 
    
    
    return Lambda, omega, Psi


def AIC_d(mu, N, n):
    #k: number of independent variables to build model. Default is k = 2.
    #L: maximum likelihood estimate of model
    #mu: Eigenvalues
    #d: Number of source signals
    
    d_list =[d for d in range(1, n)]
    # if n < max(d + 1):
    #     print('n must be greater than d')
  
    
    L_numerator = np.array([np.prod(mu[d:n]**(1/(n-d))) for d in range(1, n)])
    L_denominator = np.array([(1/(n-d))*np.sum(mu[d:n]) for d in range(1,n)])
    
    L_d = L_numerator/L_denominator
    AIC = [0 for _ in range(len(d_list))]

    for d in d_list:
        for L in L_d:
            AIC[d-1] = -2*N*(n-d)*np.emath.log(L) - 2*d*(2*n-d)
        
    k = np.argmin(AIC)
    
    return AIC, k

def GetPositiveEigs(omega):
    
    real = np.diag(omega.real)
    imag = np.diag(omega.imag)
    index = np.argwhere(real > 0)
    PositiveOmega = real[real > 0]
    
    print(f'The indices of the modes with positive eigenvalues\n are {index}')
    print(f'The magnitude of these eigenvalues \n are {PositiveOmega}')



    # Plotting:

    plt.figure()
    plt.axvline(0, c = 'black')
    plt.axhline(0, c = 'black')
    plt.title('Continuous Time Eigenvalues')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    
    plt.scatter(real, imag, c=['r' if v > 0 else 'b' for v in real])
    plt.show()
    
    return PositiveOmega