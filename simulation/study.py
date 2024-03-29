import pickle
import participant
import random
import numpy as np

class study:
    
    
    
    '''
    A population is an object which tracks participants. 
    It also tracks the total number of active days. 
    Also which participants are involved at which times. 
    '''
    
    def __init__(self,root,population_size,study_length,which_gen='case_one',sim_number = None,pop_number=0,time_condition='None'):
  
        self.root =root
        self.study_seed = sim_number*1000+1000
        self.sim_number = sim_number
        self.pop_number=pop_number
        #self.study_seed
        #8000000
        #sim_number
        self.algo_rando_gen = np.random.RandomState(seed=8000000+sim_number)
        #+pop_number
        #9000000
        #self.study_seed+1
        self.weather_gen = np.random.RandomState(seed=9000000)
        self.time_condition=time_condition
      
        with open('{}person_to_time_indices_pop_{}{}.pkl'.format(root,population_size,study_length),'rb') as f:
            pse=pickle.load(f)
        with open('{}person_to_decision_times_pop_{}{}.pkl'.format(root,population_size,study_length),'rb') as f:
            dts=pickle.load(f)
        with open('{}time_to_active_participants_pop_{}{}.pkl'.format(root,population_size,study_length),'rb') as f:
            dates_to_people=pickle.load(f)
        with open('{}all_ordered_times{}.pkl'.format(root,study_length),'rb') as f:
            study_days=pickle.load(f)
        
        self.person_to_time = pse 
        
        self.person_to_decision_times = dts 
        
        self.dates_to_people = dates_to_people 
        
        self.iters = []
        
        self.study_days = study_days 
        
        self.population = {}
    
        self.algo_seed = 2000
        
        self.history = {}
        
        self.weather_update_hours = [9,12,15,18,20]
        
        self.location_update_hours = set([9,12,16,18,20])
        
        self.update_hour = 0
        self.update_minute = 30
        self.last_update_day = study_days[0]
        self.study_length=study_length
        self.Z_one =0.1
        #,35
        self.Z_two =-0.3

        self.sigma =.35
            #.45

    
        self.init_population(which_gen,True,time_condition)
            
    def get_gid(self):
       
         return int(random.random()>=.5)+1
    
    def update_beta(self,features):
        
        self.beta =np.array([  0.05,  -0.15,  0.2,  0.3, -0.3 ])
        
        potential_features = ['intercept','tod','dow','pretreatment','location']
        new = np.array([self.beta[0]]+[self.beta[i] for i in range(len(self.beta)) if potential_features[i] in features])

        self.regret_beta = new
    
    
    
    
    def init_population(self,which_gen,location,time_condition='None'):
         
        for k,v in self.person_to_time.items():
            
          
            

            #k+self.sim_number*1000
            person_seed =k+100
                #k+self.pop_number*1000

            rg=np.random.RandomState(seed=person_seed)
            
            gid = int(rg.uniform()>=.5)+1

            
            Z=None
            if which_gen=='case_two':
                Z = self.Z_one
                if gid==2:
                    Z=self.Z_two
            if which_gen=='case_three':
                                        
                Z=rg.normal(loc=0,scale=self.sigma)
            
            
         
         #this_beta = [i for i in [  0.05,  0.25,  0.25,  0.25, -0.3]]
         #this_beta = [i for i in [  0.05,  0.25,  0.25,  -0.3, 0.25]]
         #this_beta = [0.05,  0.25,  0.2,  -.1, -.35]
         #this_beta=[0.05,  -0.1,  0.1,  0.3, -0.35]
            this_beta=[0.05,  -0.15,  0.2,  0.3, -0.3   ]
            if location:
                if which_gen=='case_two':
                    offset = .1
                    if gid==2:
                        offset = offset*-1
                    this_beta[-1]=this_beta[-1]+offset
                if which_gen=='case_three':
                    
                    l=rg.normal(loc=0,scale=0.1)
                    this_beta[-1]=this_beta[-1]+l
        
            beta_regret = [i for i in this_beta]
            if time_condition=='no_location':
                this_beta[2] = .8
                beta_regret = this_beta[:2]+this_beta[3:]
            if time_condition=='burden':
                #this_beta[2] = .35
                beta_regret = this_beta
                #this_beta = [.35,.2,.15,.1,0.05,0.0,-.1,-.2,-.3,-.4,-.5,-.6]+this_beta[1:]
                #short good:[.2,.1,0.00,-0.05,-.15,-.25]
                this_beta = [0.25 ,  0.2, .15, 0.1,  0.05 ,   0.  ,  -0.1 , -0.15,-0.2 , -.25]+this_beta[1:]
                
            person = participant.participant(pid=k,gid=gid,times=v,decision_times = self.person_to_decision_times[k],Z=Z,rg=rg,beta=np.array(this_beta),beta_regret = beta_regret)
     
            self.population[k]=person



