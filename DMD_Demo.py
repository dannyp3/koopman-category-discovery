import numpy as np
import matplotlib.pyplot as plt


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
    #omega[omega == -np.inf] = 0
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








#########################################################################
# Everything below this line is stuff to look at for the demo


import random

# Generate n random integers between 30 MHz and 300 MHz
n = 1000
X = np.array([random.randint(30E6, 300E6) for i in range(n)])
X = np.reshape(X, (100, 10)) # reshape to feed into DMD algorithm


t = np.arange(0, np.shape(X)[1])
Phi, A = DMD(X, 9)
Lambda, omega, Psi = DMD_Recon(X, 9, t)

GetPositiveEigs(omega) # Creates plot of continuous time eigenvalues (real and imaginary axes)
plt.plot(t[:9], omega[0,:]) # Plot of eigenvalue 1
plt.show()
plt.plot(t[:9], omega[1,:]) # Plot of eigenvalue 2 
plt.show()
plt.plot(t[:9], omega[2,:]) # Plot of eigenvalue 3
plt.show()
plt.plot(t[:9], omega[3,:]) # Plot of eigenvalue 4
plt.show()
plt.plot(t[:9], omega[4,:]) # Plot of eigenvalue 5
plt.show()
plt.plot(t[:9], omega[5,:]) # Plot of eigenvalue 6
plt.show()
plt.plot(t[:9], omega[6,:]) # Plot of eigenvalue 7
plt.show()
plt.plot(t[:9], omega[7,:]) # Plot of eigenvalue 8 
plt.show()
plt.plot(t[:9], omega[8,:]) # Plot of eigenvalue 9
plt.show()

for k in range(np.shape(omega)[0]):
    plt.plot(t[:9], omega[k,:]) # Plot all the eigenvalues at once

plt.plot(t[:9], np.diag(omega))