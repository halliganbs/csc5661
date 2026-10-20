import numpy as np
import itertools

import copy

import seaborn as sb

from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#two classes for these two environments
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class race_environment:
    '''A class to manage the race'''
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def __init__(self, config):
        '''Set up the constructor
            Takes -- config, a dictionary specifying the track dimensions and initial state
        '''
        self.track = np.zeros((config['track_shape']['height']
                  ,config['track_shape']['width']))    #set up an array to represent the track
        self.track[config['track_shape']['v_track_width']:,config['track_shape']['h_track_width']:] = 1    #block the no-go area
        self.state = {}
        if not config['starting_position']:
            self.state['position'] = {'X':np.random.choice(a = range(config['track_shape']['v_track_width']))
                                      ,'Y':config['track_shape']['height']-1}
        else:
            self.state['position'] = copy.deepcopy(config['starting_position'])
        if not config['starting_velocity']:
            self.state['V'] = {'V':0,'H':0}
        else:
            self.state['V'] = copy.deepcopy(config['starting_velocity'])    #set up the state -- here it will be position and velocity
        self.old_position = copy.deepcopy(self.state['position'])
        self.config = config        
        
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def reset(self):
        '''A function to reset to initial conditions'''
        self.state = {'position':{'X':np.random.choice(a = range(self.config['track_shape']['v_track_width']))
                                      ,'Y':self.config['track_shape']['height']-1}
                      ,'V':{'V':0,'H':0}}    #set up the state -- here it will be position and velocity
        self.old_position = copy.deepcopy(self.state['position'])

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def update_V(self,a):
        '''A function to update vert and horiz velocity
            Takes -- a, a dictionary containing the vert and horiz acceleration
        '''
        self.state['V']['V'] = min(5,max(1,self.state['V']['V'] + a['V']))    #update the vert velocity
        self.state['V']['H'] = min(5,self.state['V']['H'] + a['H'])    #update the horiz velocity

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def update_Position(self):
        '''A function to update vert and horiz position
        '''        
        self.state['position']['X'] = self.state['position']['X'] + self.state['V']['H']    #update the horiz position
        self.state['position']['Y'] = self.state['position']['Y'] - self.state['V']['V']    #update the vert position

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def obstacle_check(self):
        '''A function to check if the player crosses an obstacle'''
        crash = False
        if self.state['position']['X'] != self.old_position['X']:
            m = (self.state['position']['Y'] - self.old_position['Y'])/(self.state['position']['X'] - self.old_position['X'])
            b = self.state['position']['Y'] - m*self.state['position']['X']
            xs = np.linspace(start = self.old_position['X']
                        ,stop = self.state['position']['X']
                        ,num = 2*abs(self.state['position']['X'] - self.old_position['X']) + 1)
            ys = np.round(m*xs + b)
            xs = np.round(xs)
            for x,y in zip(xs,ys):

                if self.track[int(y),int(x)] == 1:
                    crash = True
                    break
        else:
            if self.track[self.state['position']['Y'],self.state['position']['X']] == 1:    #and if the agent is in the no-go
                crash = True    #set crash to true
        return crash

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def check_crash(self):
        '''A function to check if the agent crashed'''
        crash = False
        if self.state['position']['X'] < 0:    #if the agent goes out of bounds to the left
            crash = True    #set crash to true
        if self.state['position']['Y'] < 0:    #if the agent goes out of bounds to the top
            crash = True    #set crash to true
        if self.state['position']['X'] >= 0 and self.state['position']['X'] < self.config['track_shape']['width']-1 and self.state['position']['Y'] >= 0:    #if the agent is within the array
            crash = self.obstacle_check()
        return crash

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def check_win(self):
        '''A function to check if the agent won'''
        win = False    #assume no win
        if self.state['position']['X'] >= self.config['track_shape']['width'] and self.state['position']['Y'] < self.config['track_shape']['h_track_width']:    #if the agent is over the finish line
            win = True    #set win to true
        return win

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def reward(self,sp):
        return 1/max(1,((sp['X'] - (self.config['track_shape']['width'] - 1))**2
                          + (self.config['track_shape']['h_track_width']/2.0 - sp['Y'])**2))
        
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def step(self,a):
        '''A function to evolve the environment
            Takes -- a, a dictionary containing the vert and horiz acceleration
        '''
        outcome = ''    #track crash/win
        done = False    #set the done indicator
        self.update_V(a)    #update the velocity
        self.update_Position()    #update the position
        crash = self.check_crash()    #check if the agent crashed
        self.old_position = copy.deepcopy(self.state['position'])
        r = self.reward(self.state['position'])
        if crash:
            outcome = 'crash'
            r = -1
            self.reset()
        if not crash:    #if the agent didn't crash...
            win = self.check_win()    #check if the agent won
            if win:
                outcome = 'won'
                done = True
                r = 10
        return {'state':self.state
                     ,'reward':r
                     ,'done':done
                    ,'Win/Lost':outcome}

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#two classes for these two environments
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Agent:
    '''A class to manage the agent'''
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def __init__(self, config):
        '''Set up the constructor
            Takes -- config, a dictionary specifying the track dimensions and initial state
        '''
        self.Z = {} # state action pairs
        self.Q = {} # blank Q-values
        self.config = config

    def setup_Q(self, track_shape):
        # creates all state action pairs, and sets to zero for Q values and Z
        for x in range(0, track_shape['width']):
            for y in range(0, track_shape['height']):
                s = f"{y}_{x}"
                self.Q[s] = {}
                self.Z[s] = {}
                for a1, a2 in self.config['A']:
                    a = f"{a1}_{a2}"
                    self.Q[s][a] = 0
                    self.Z[s][a] = []

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def update_Q(self,state, action, value):
        self.Q[state][action] = value
        
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
    def pi(self,state,epsilon):
        actions = self.Q[state] # get q-values for each action
        max_act = max(actions, key=actions.get)
        if self.Q[state][max_act] >= (epsilon):
            v, h = max_act.split('_')
            action = {
                'V':int(v),
                'H':int(h)
            }
        else:
            act = np.random.choice(list(actions.keys()))
            v, h = act.split('_')
            action = {
                'V':int(v),
                'H':int(h)
            }
        return action
    
def to_state(state):
    # {'position': {'X': np.int64(3), 'Y': 29}, 'V': {'V': 0, 'H': 0}} to string
    x = state['position']['X']
    y = state['position']['Y']
    return f"{y}_{x}"

height = 30    #set track height
width = 25    #set track width
v_track_width = 10    #set road width for the vert section
h_track_width = 10    #set road width for the horiz section
env_config = {'track_shape':{'height':height
                              ,'width':width
                              ,'v_track_width':v_track_width
                              ,'h_track_width':h_track_width}
          ,'starting_position':None
         ,'starting_velocity':None}    #set up the config for the environment

# agent_config = {'epsilon':0.1,'gamma':0.9,'A':list(itertools.product([-1,0,1],[-1,0,1]))}    #set up the config for the agent

# SARSA w/ Greedy

renv = race_environment(env_config)
agent_config = {'epsilon':0.4,'gamma':0.5,'A':list(itertools.product([-1,0,1],[-1,0,1]))}
agent = Agent(agent_config)
agent.setup_Q(env_config['track_shape'])

epsilon = agent_config['epsilon']
gamma = agent_config['gamma']
alpha = 0.7

N = 5000 # number episodes

state = to_state(renv.state)
action = agent.pi(state, epsilon)


runtime_sarsa={
    'episode':[],
    'T':[],
    'num_crash':[],
    'ep_reward':[]
}

for n in range(N):
    T = 0
    avg_r = 0
    c = 0
    done = False
    while not done:
        a_c = f"{action['V']}_{action['H']}"
        # sample actions according to pi
        rewards = renv.step(action) # implement action to generate r'
        r = rewards['reward']
        if not rewards['done']:
            new_state = to_state(rewards['state'])
            new_action = agent.pi(state, epsilon) # sample a' according to pi(s')
            a_n = f"{new_action['V']}_{new_action['H']}" 
            
            # update Q values
            new_Q = agent.Q[state][a_c] + alpha*(r + gamma*agent.Q[new_state][a_n] - agent.Q[state][a_c])
            agent.update_Q(state,a_c, new_Q)
            action = new_action
            state = new_state
            
        avg_r += rewards['reward']
        if rewards['Win/Lost'] == 'crash':
            c+=1
        
        T+=1   
        done = rewards['done']
        
        # clear_output(wait=True)
        print(n, T, rewards)
        
    runtime_sarsa['episode'].append(n)
    runtime_sarsa['T'].append(T)
    runtime_sarsa['num_crash'].append(c)
    runtime_sarsa['ep_reward'].append(avg_r)
    renv.reset()

sarsa_data = pd.DataFrame(data=runtime_sarsa)
sarsa_data.to_csv('sarsa_rewards.csv', index=False)