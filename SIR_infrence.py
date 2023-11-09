
import nbformat
import pyprob
from pyprob import Model
from pyprob.distributions import Normal, Uniform, Exponential
import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure();


# |%%--%%| <jet0sZSpdO|PZF6FUuCLr>

class SIRModel(Model):
    def __init__(self, N, I0):
        super().__init__() 
        self.N = N
        self.I0 = I0
       
        self.prior_mean = 1
        self.prior_std = math.sqrt(5)
        self.likelyhood_std = 5

    def forward(self):
         #INITIALISE VARIABLES
    
        #time variables. 
        t = 0 #total time that has elapsed 
        t1 = 0 #Time of Expovariate with parameter rate of transmission
        t2 = 0 #Time of Expovariate with parameter rate of recoveries
        t3 = 0 #lowest value of t1 or t2 

        #the current S, I, R variables 
        Sn = self.N - self.I0
        In = self.I0
        Rn = 0

        #list of t, S, I, R to output for plot
        t_list = [t]
        S_list = [Sn]
        I_list = [In]
        R_list = [Rn]
        

        beta = pyprob.sample(Normal(self.prior_mean, self.prior_std))
        gamma = pyprob.sample(Normal(self.prior_mean, self.prior_std))
        
        count = 0
        while In > 0 and Sn > 0: #while a person is still infected
            beta = abs(beta)
            gamma = abs(gamma)
            
            rate_of_transmission = (beta * Sn * In) / self.N #rate if the person transmits the infection
            rate_of_recovery = gamma * In #rate if the person recovers from the infection
            
            t1 = pyprob.sample(Exponential(rate_of_transmission))
            t2 = pyprob.sample(Exponential(rate_of_transmission))
            
            t3 = min(t1.item(), t2.item()) #finds out which of the two events (t1, t2) happens first 

            t += t3 #add the event that happens to total time 
            t_list += [t] #add the event time to the times list 

            if t3 == t1: #if transmission happens first, the new state is (S − 1, I + 1, R)
                Sn -= 1
                In += 1

            if t3 == t2: #if recovery happens first, the new state is (S, I − 1, R + 1)
                In -= 1
                Rn += 1

            #Add current values of S,I,R, into a list
            S_list += [Sn]
            I_list += [In]
            R_list += [Rn]
            
            pyprob.observe(Normal(Sn, self.likelyhood_std), name="sus" + str(count))
            pyprob.observe(Normal(In, self.likelyhood_std), name="inf" + str(count))
            pyprob.observe(Normal(Rn, self.likelyhood_std), name="rec" + str(count))
            
            count += 1
        
        return beta, gamma, t1, t2 #output the times, S,I,R list 

# |%%--%%| <PZF6FUuCLr|nEE1ZSRkwP>

sirModel = SIRModel(1000, 10)

# |%%--%%| <nEE1ZSRkwP|iXDoAh0IJS>

def observant_generator(N,I0, beta, gamma):
    #INITIALISE VARIABLES
    
    #time variables. 
    t = 0 #total time that has elapsed 
    t1 = 0 #Time of Expovariate with parameter rate of transmission
    t2 = 0 #Time of Expovariate with parameter rate of recoveries
    t3 = 0 #lowest value of t1 or t2 
    
    #the current S, I, R variables 
    Sn = N - I0
    In = I0
    Rn = 0
    
    #list of t, S, I, R to output for plot
    t_list = [t]
    S_list = [Sn]
    I_list = [In]
    R_list = [Rn]
    
    while In > 0 and Sn > 0: #while a person is still infected
        beta = abs(beta)
        gamma = abs(gamma)
        rate_of_transmission = (beta * Sn * In) / N #rate if the person transmits the infection
        rate_of_recovery = gamma * In #rate if the person recovers from the infection
        
        t1 = random.expovariate(rate_of_transmission) #Exponentially distributed R.V. with rate of transmission
        t2 = random.expovariate(rate_of_recovery) #Exponentially distributed R.V. with rate of recovery
        t3 = min(t1, t2) #finds out which of the two events (t1, t2) happens first 
        
        t += t3 #add the event that happens to total time 
        t_list += [t] #add the event time to the times list 
        
        if t3 == t1: #if transmission happens first, the new state is (S − 1, I + 1, R)
            Sn -= 1
            In += 1
            
        if t3 == t2: #if recovery happens first, the new state is (S, I − 1, R + 1)
            In -= 1
            Rn += 1
            
        #Add current values of S,I,R, into a list
        S_list += [Sn]
        I_list += [In]
        R_list += [Rn]
        
    return t_list, S_list, I_list, R_list #output the times, S,I,R list 

# |%%--%%| <iXDoAh0IJS|B9YB09J4IW>

observdic = {}
for x in range(5000):
    b = np.random.normal(3, 0.3)
    g = np.random.normal(2, 0.3)
    T, S, I, R = observant_generator(1000, 10, b, g)
    for y in range(len(S)):
        address = "sus" + str(y)
        if address in observdic:
            s = observdic[address]
            s.append(S[y])
            observdic[address] = s
            i = observdic["inf"+str(y)]
            i.append(I[y])
            observdic["inf"+str(y)] = i
            r = observdic["rec"+str(y)]
            r.append(R[y])
            observdic["rec"+str(y)]
        else:
            observdic[address] = [S[y]]
            observdic["inf"+str(y)] =  [I[y]]
            observdic["rec"+str(y)] =  [R[y]]

print(len(observdic["inf0"]))

# |%%--%%| <B9YB09J4IW|cGv1kID9sP>

posterior = sirModel.posterior_results(
                    num_traces=7000, # the number of samples estimating the posterior
                    inference_engine=pyprob.InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS, 
                    observe= observdic )

# |%%--%%| <cGv1kID9sP|gpEIncTR10>

print(posterior.sample())

# |%%--%%| <gpEIncTR10|5XmRvKheSn>

posterior_first = posterior.map(lambda v: v[0]) 
print("mean for beta =",posterior_first.mean)
posterior_second = posterior.map(lambda v: v[1])
print("mean for gamma =",posterior_second.mean)
posterior_third = posterior.map(lambda v: v[2])
print("mean for inf_time =",posterior_second.mean)
posterior_fourth = posterior.map(lambda v: v[3]) 
print("mean for heal_time =",posterior_second.mean)

# |%%--%%| <5XmRvKheSn|UQ4Nt2cePk>

posterior_first.plot_histogram(show=True, bins=100)

# |%%--%%| <UQ4Nt2cePk|YjmQ1CUq9L>

posterior_second.plot_histogram(show=True, bins=100)

# |%%--%%| <YjmQ1CUq9L|5yy0SLdEwC>

posterior_third.plot_histogram(show=True, bins=100)

# |%%--%%| <5yy0SLdEwC|j5isJIfBMt>

posterior_fourth.plot_histogram(show=True, bins=100)

# |%%--%%| <j5isJIfBMt|YQeEdXDjPJ>
r"""°°°
Below is the inference compilation
°°°"""
# |%%--%%| <YQeEdXDjPJ|ZO4thZelZ8>

sirModel.learn_inference_network(num_traces=20000,
            observe_embeddings=observemb,
            inference_network=pyprob.InferenceNetwork.LSTM)

# |%%--%%| <ZO4thZelZ8|5H8KID1e3M>

inference_posterior = sirModel.posterior_results(
                num_traces=10000, # the number of samples estimating the posterior
                inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, 
                observe= observdic )

# |%%--%%| <5H8KID1e3M|tkam72zOvH>

inf_posterior_first = posterior.map(lambda v: v[0]) 
print("mean for beta =",posterior_first.mean)
inf_posterior_second = posterior.map(lambda v: v[1])
print("mean for gamma =",posterior_second.mean)
inf_posterior_third = posterior.map(lambda v: v[2])
print("mean for inf_time =",posterior_second.mean)
inf_posterior_fourth = posterior.map(lambda v: v[3]) 
print("mean for heal_time =",posterior_second.mean)

# |%%--%%| <tkam72zOvH|ZVZC0whL5F>

inf_posterior_first.plot_histogram(show=True, bins=100)

# |%%--%%| <ZVZC0whL5F|YNTLURCgE4>

inf_posterior_second.plot_histogram(show=True, bins=100)

# |%%--%%| <YNTLURCgE4|UmJl3tzYkt>

inf_posterior_third.plot_histogram(show=True, bins=100)

# |%%--%%| <UmJl3tzYkt|767YLA13aU>

inf_posterior_fourth.plot_histogram(show=True, bins=100)
