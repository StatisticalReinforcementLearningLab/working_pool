import pandas as pd
import numpy as np
import pickle
import random
import os
import math
import run_gpytorchkernel
import run_gpytorchkernel_larger
import operator
import study
import time as time_module
import simple_bandits
import TS_personal_params_pooled as pp
import TS_global_params_pooled as gtp
from numpy.random import uniform
import TS
import warnings
warnings.simplefilter('ignore')
from sklearn import preprocessing
import tensorflow as tf
import gc
import feature_transformations as ft
import run_hob


def update(algo_type,train_type,experiment,time,global_policy_params,personal_policy_params,feat_trans,participant=None):



    if algo_type=='batch' or algo_type=='pooling' or algo_type=='pooling_four':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
        temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
            
        steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)

        if algo_type=='batch':
            ##
            temp = TS.policy_update_ts_new( temp_data[0],steps,probs,actions,global_policy_params.noise_term,\
                                           global_policy_params.mu1_knot,\
                                           global_policy_params.sigma1_knot,\
                                           global_policy_params.mu2_knot,\
                                           global_policy_params.sigma2_knot)
                
                
            mu_beta = temp[0]
            Sigma_beta = temp[1]

            global_policy_params.update_mus(None,mu_beta,2)
            global_policy_params.update_sigmas(None,Sigma_beta,2)
            for participant in experiment.population.values():
                
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                participant.last_update_day=time

        elif algo_type=='pooling':
            try:
                temp_params = run_gpytorchkernel.run(temp_data[0], temp_data[1],steps,global_policy_params)
                
                experiment.iters.append(temp_params['iters'])
                
                if temp_params['cov'] is not None:
                    global_policy_params.update_params(temp_params)
                        
            except Exception as e:
                temp_params={'cov':global_policy_params.cov,\
                                'noise':global_policy_params.noise_term,\
                                    'like':-100333,'sigma_u':global_policy_params.sigma_u}
            inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)
                                            
            global_policy_params.inv_term = inv_term
            global_policy_params.history = temp_data
            for participant in experiment.population.values():
                if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                    temp = simple_bandits.calculate_posterior_faster(global_policy_params,\
                                                                              participant.pid,participant.current_day_counter,\
                                                                              global_policy_params.history[0], global_policy_params.history[1],global_policy_params.history[2] )
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                    ##change here
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)

                    participant.last_update_day=time
                
                
        elif algo_type=='pooling_four':
            try:
            
            
                temp_params = run_gpytorchkernel_larger.run(temp_data[0], temp_data[1],steps,global_policy_params)
            

                        #print(temp_params)
#global_policy_params.history =temp_data
                if temp_params['cov'] is not None:
                    global_policy_params.update_params_more(temp_params)
                inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)
                        
                global_policy_params.inv_term=inv_term
                        #print(temp_params)
                global_policy_params.history =temp_data
            except Exception as e:
                print('something failed')
                print(e)
                pass
                    

                        
            global_policy_params.history =temp_data
            for participant in experiment.population.values():
                if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                    temp = simple_bandits.calculate_posterior_current(global_policy_params,\
                                                                 participant.pid,participant.current_day_counter,\
                                                                 global_policy_params.history[0], global_policy_params.history[1],global_policy_params.history[2] )
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                ##change here
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
    
                    participant.last_update_day=time
            
            
    
    elif algo_type=='personalized':
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
        
                temp_hist = feat_trans.get_history_decision_time_avail_single({participant.pid:participant.history},time)
                temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
                context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
                temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
                context = temp_data[0]
                steps = feat_trans.get_RT_o(steps,temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
                temp = TS.policy_update_ts_new( context,steps,probs,actions,global_policy_params.noise_term,\
                                           global_policy_params.mu1_knot,\
                                           global_policy_params.sigma1_knot,\
                                           global_policy_params.mu2_knot,\
                                           global_policy_params.sigma2_knot,
                                           
                                           
                                           )
                mu_beta = temp[0]
                Sigma_beta = temp[1]
                
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                participant.last_update_day=time


    elif algo_type=='hob':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,users,steps= feat_trans.get_hob_form(temp_hist,global_policy_params)
        learned = run_hob.update_params(global_policy_params,context,steps)
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                my_vec = learned[participant.pid*global_policy_params.d:participant.pid*global_policy_params.d+global_policy_params.d][-(global_policy_params.num_responsivity_features+1):]
                #print(learned.shape)
                #print(my_vec)
                personal_policy_params.update_mus(participant.pid,my_vec,2)

    elif algo_type=='hob_clipped':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,users,steps= feat_trans.get_hob_form_clipped(temp_hist,global_policy_params)
        mu,sigma = run_hob.update_params_clipped(global_policy_params,context,steps)
        M=global_policy_params.d

        blocks = [sigma[i*M:(i+1)*M,i*M:(i+1)*M] for i in range(int(sigma.shape[0]/M))]
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                my_vec = mu[participant.pid*global_policy_params.d:participant.pid*global_policy_params.d+global_policy_params.d][-(global_policy_params.num_responsivity_features+1):]
                
                my_sigma = [j[-(global_policy_params.num_responsivity_features+1):] for j in blocks[participant.pid][-(global_policy_params.num_responsivity_features+1):]]
            #print(learned.shape)
            #print(my_vec)
            
            ##my mats
            
                personal_policy_params.update_mus(participant.pid,my_vec,2)





    else:
        return 'Excepted types are batch, pooled, personalized, pooled_four, or hob'







