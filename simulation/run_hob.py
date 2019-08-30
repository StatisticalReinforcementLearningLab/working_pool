import numpy as np
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sps
from scipy.sparse.linalg import cg
import scipy

def get_covariance(large_user_graph,context):
    z = np.matmul(np.array(context).T,np.array(context))
    #print(z.shape)
    #500000
    #print(np.array(context).std())
    #.01
    #.2
    #(1/(.9))
    return (1/3)*z+1*large_user_graph

def get_prior(glob):
    temp = np.random.multivariate_normal(np.zeros(glob.d*glob.users),np.eye(glob.d*glob.users))
    #big_S = np.kron(scipy.linalg.cholesky(L),np.eye(d))
    x = cg(glob.big_S,temp)[0]
    while np.isnan(x).any():
        temp = np.random.multivariate_normal(np.zeros(glob.d*glob.users),np.eye(glob.d*glob.users))
        #big_S = np.kron(scipy.linalg.cholesky(L),np.eye(d))
        x = cg(glob.big_S,temp)[0]
    return x

def b_term(glob,context_matrix,reward_vector):
    prior = get_prior(glob)
    #print(prior)
    first_term = np.matmul(glob.lu,prior)
    b_t = np.matmul(np.array(context_matrix).T,np.array(reward_vector))
    decision_times = context_matrix.shape[0]
    noise_vector = np.random.multivariate_normal(np.zeros(decision_times),np.eye(decision_times))
    last_term = np.matmul(np.array(context_matrix).T,np.array(noise_vector))
    
    return first_term+b_t+last_term

def update_params(glob_p,context,steps):
    #print(context.shape)
    cov = get_covariance(glob_p.lu,context)
    b = b_term(glob_p,context,steps)
    y = cg(cov,b)[0]
    return y

def get_mean(glob,context_matrix,reward_vector,inv_sigma):
    b_t = np.matmul(np.array(context_matrix).T,np.array(reward_vector))
    ##change to global sigma
    ##
    return np.matmul(inv_sigma,b_t)

def get_person_blocks(global_params):
    
    #[j[-(global_params.num_responsivity_features+1):] for j in sigma[-(global_params.num_responsivity_features+1):]]
    pass

def update_params_clipped(glob_p,context,steps):
    #print(context.shape)
    cov = get_covariance(glob_p.lu,context)
    ##INVERT
    cov = np.linalg.inv(cov)
    mean  = get_mean(glob_p,context,steps,cov)
    return mean,cov

