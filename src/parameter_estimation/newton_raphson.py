import numpy as np
import math




corr_functions_dict = {
    "matern": corr_matern,
    "exponential": corr_exponential,
    "gaussian": corr_gaussian,
    "matern_3_2": corr_matern_3_2,
    "matern_5_2": corr_matern_5_2
}


def log_likelihood(y, mu, K, tau, sigma):
    n = len(y)
    b = y - mu
    Sigma = K + np.eye(n) * tau**2
    L = np.linalg.cholesky(Sigma)
    determinant = 2 * np.sum(np.log(np.diag(L))) 
    return - 1 / 2 * ( b @ Sigma @ b.T + determinant )


def log_determinant(K, tau, sigma):
    n = len(K)
    L = np.linalg.cholesky(K + np.eye(n) * tau**2)
    determinant = 2 * np.sum(np.log(np.diag(L))) + n * np.log(sigma ** 2) 
    return determinant

def sigma_hat(y, mu, K, tau, inv_matrix=None):
    n = len(y)
    if inv_matrix is None:
        A = np.linalg.inv(K + np.eye(n) * tau**2)
    else:
        A = inv_matrix
    b = y - mu
    return 1 / n * b @ A @ b.T


def ml_itrative_estimate(y, mu):

    pass

def dSigma_dPhi(model_dict, corr_type, domain):
    K = model_dict["K"]
    D = model_dict["D"]
    phi = model_dict["phi"]
    if domain == "temporal":
        D = model_dict["Dt"]
        phi = model_dict["phi_t"]
    if domain == "spatial":
        D = model_dict["Ds"]
        phi = model_dict["phi_s"]

    if corr_type == "gaussian":
        return dSigma_dPhi_gaussian(K, D, phi)
    if corr_type == "exponential":
        return dSigma_dPhi_exp(K, D, phi)
    if corr_type == "matern 3/2":
        pass


def est_beta_hat(y, mu, K, tau, sigma):


    Q = np.linalg.inv(K + np.eye(len(y)) * tau**2)
    H = np.array([np.repeat(1, len(y)), mu]).T  # Design matrix

    print("H shape", H.shape)
    HT_Q = H.T @ Q

    beta = np.linalg.inv(HT_Q @ H) @ HT_Q @ y
    return beta


def dSigma_dTau2(y):
    # The derivative of Sigma with repsect to tau^2
    n = len(y)
    return np.eye(n)

def dSigma_dsigma2(K, sigma2):
    return K / sigma2

def dSigma_dPhi_s(K, Ds):
    # The derivative of Sigma with respect to phi_s
    return - Ds**2 / 2 * K 

def dSigma_dPhi_t(K, Dt):
    # The derivative of Sigma with respect to phi_t
    return - Dt**2 / 2 * K


def dSigma_dPhi_exp(K, D , phi=1):
    # The derivative of Sigma with respect to phi_t
    return - D / 2 * K


def dSigma_dPhi_gaussian(K, D, phi=1):
    # The derivative of Sigma with respect to phi_t
    return - D / 2 * K

def dSigma_dPhi_matern(K, D, phi):
    # The derivative of Sigma with respect to phi_t
    return - D * phi**2 * K









    




def hessian(Q, dSigma_dsigma2, dSigma_dTau2, dSigma_dPhi_s, dSigma_dPhi_t):

    # Thes are repeated calculations
    Q_dSigma_dsigma2 = Q @ dSigma_dsigma2
    Q_dSigma_dTau2 = Q @ dSigma_dTau2
    Q_dSigma_dPhi_s = Q @ dSigma_dPhi_s
    Q_dSigma_dPhi_t = Q @ dSigma_dPhi_t

    # The hessian is a 4x4 matrix
    hessian = np.zeros((4, 4))

    Q_der_list = [Q_dSigma_dsigma2, Q_dSigma_dTau2, Q_dSigma_dPhi_s, Q_dSigma_dPhi_t]
    for i in range(4):
        for j in range(4):
            hessian[i, j] = -1/2 * np.trace(Q_der_list[i] @ Q_der_list[j])
    
    return hessian


def score(Q, Z, dSigma_dsigma2, dSigma_dTau2, dSigma_dPhi_s, dSigma_dPhi_t):
            
    Q_dSigma_dsigma2 = Q @ dSigma_dsigma2
    Q_dSigma_dTau2 = Q @ dSigma_dTau2
    Q_dSigma_dPhi_s = Q @ dSigma_dPhi_s
    Q_dSigma_dPhi_t = Q @ dSigma_dPhi_t

    score = np.zeros(4)

    Q_der_list = [Q_dSigma_dsigma2, Q_dSigma_dTau2, Q_dSigma_dPhi_s, Q_dSigma_dPhi_t]

    ZT_Q = Z.T @ Q
    QZ = ZT_Q.T
    QZ = Q @ Z

    for i in range(4):
        part_I = - 1/2 * np.trace(Q_der_list[i])
        part_II = 1/2 * ZT_Q @ Q_der_list[i] @ QZ
        score[i] = part_I + part_II

    return score



def newton_raphson_iteration(y, mu, K, Ds, Dt, sigma2, tau2, phi_s, phi_t, epsilon=1):


    # Define the initial values
    n = len(y)
    Sigma = K + np.eye(n) * tau2
    Q = np.linalg.inv(Sigma)

    # Compute Beta_hat first
    H = np.array([np.repeat(1, n), mu]).T  # Design matrix
    #print("H shape", H.shape)
    #print("H", H)
    HT_Q = H.T @ Q
    beta_hat = np.linalg.inv(HT_Q @ H) @ HT_Q @ y

    # Then define Z
    Z = y - H @ beta_hat

    # Compute the derivatives
    dSigma_dtau2 = np.eye(n)
    dSigma_dsigma2 = K / sigma2
    dSigma_dPhi_s = - Ds**2 * K  * 0.5
    dSigma_dPhi_t = - Dt**2 * K * 0.5

    # Compute score and hessian
    score_ = score(Q, Z, dSigma_dsigma2, dSigma_dtau2, dSigma_dPhi_s, dSigma_dPhi_t)
    hessian_ = hessian(Q, dSigma_dsigma2, dSigma_dtau2, dSigma_dPhi_s, dSigma_dPhi_t)

   

    # Compute the new values
    old_values = np.array([sigma2, tau2, phi_s, phi_t])
    new_values = old_values - np.linalg.inv(hessian_) @ score_ * epsilon 

    print("beta_hat", beta_hat)    
    print("Score", score_)
    print("Hessian", hessian_)
    print("old values", old_values)
    print("new values", new_values)

    return new_values, beta_hat
     

if __name__== "__main__":

    y = np.array([1, 2, 3, 4, 5])
    mu = np.array([1, 2.2, 3.3, 4.4, 5.5]) - 10
    K = np.eye(5)
    tau = 0.1
    sigma = 0.1
    print(log_likelihood(y, mu, K, tau, sigma))
    print(log_determinant(K, tau, sigma))
    print(sigma_hat(y, mu, K, tau))
    beta = est_beta_hat(y, mu, K, tau, sigma)

    print(beta)


    # Plot some correlation functions

    import matplotlib.pyplot as plt
    h = np.linspace(0, 1000, 100)
    plt.plot(h, corr_gaussian(h, 6.97602437e-03), label="Gaussian")
    plt.plot(h, corr_exponential(h, 2.96067809e-06), label="Exponential")   
    plt.show()