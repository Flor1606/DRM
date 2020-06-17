import gym
from gym import error, spaces, utils
from gym.utils import seeding
class DRM(gym.Env):
    #metadata = {'render.modes': ['human']} 
    def __init__(self):
        #variables del entorno
        self.t = 11
        self.f = 19.7 #GHz
        self.G_T = 17 #dB/K
        self.Loss = 20*math.log10((4*3.1416*36000000)/(0.38/self.f)) #perdidas espacio libre dB
        self.BdW_b_min = 100 #Ancho de banda minimo
        self.BdW_b_max = 500 #Ancho de banda maximo
        self.P_b_min = 8 #Potencia minima
        self.P_b_max = 15 #Potencia maxima
        self.BdW_b = np.linspace(self.BdW_b_min, self.BdW_b_max, num=9) #MHz
        self.P_b = np.linspace(self.P_b_min, self.P_b_max, num=50) #dBW
        self.BW_1 = 0.65
        self.BW_2 = 0.60
        self.BW_3 = 0.55
        self.BmW_b = np.array([self.BW_1, self.BW_2, self.BW_3])
        self.NBdW = len(self.BdW_b)
        self.NP = len(self.P_b)
        self.NBmW = len(self.BmW_b)
        self.C_b = np.zeros((self.NBmW, self.NBdW, self.NP)) # Matriz de ceros
                
        self.P_b_t = 15
        self.BmW_b_t = 100
        self.BdW_b_t = 0.6
        #self.stateSpace = [i for i in range(self.NBdW*self.NBdW*self.NP)]
        #self.stateSpacePlus = [i for i in range(self.NBdW*self.NBdW*self.NP)]
        #acciones
        self.action_space = {'-P_b': -1, '+P_b': 1, '-BdW_b': -2, '+BdW_b': 2,
                            '-BmW_b': -3, '+BmW_b': 3, 'nothing': 0}
        self.possibleActions = [-1, 1, -2, 2, -3, 3, 0]
        #self.possibleActions = ['-P_b', '+P_b', '-BdW_b', '+BdW_b', '-BmW_b','+BmW_b','nothing']
        
        #self.P_b_t = random.choice(self.P_b)
        #self.BmW_b_t = random.choice(self.BmW_b)
        #self.BdW_b_t = random.choice(self.BdW_b)
        
        self.agentPositionP1 = np.where(self.P_b == self.P_b_t)
        self.agentPositionP = self.agentPositionP1[0]
        self.agentPositionBdW1 = np.where(self.BdW_b == self.BdW_b_t) 
        self.agentPositionBdW = self.agentPositionBdW1[0]
        self.agentPositionBm1 = np.where(self.BmW_b == self.BmW_b_t)
        self.agentPositionBmW = self.agentPositionBm1[0]
        
        self.observation_space = [i for i in range(self.NBdW*self.NBdW*self.NP)]
        
        #self.seed()
        #self.reset()
        
        def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]
        
        def isTerminalState(self, state):
            return state in self.stateSpacePlus and state not in self.stateSpace
        
        def CbSpace(self):
        #print(C_b)
            for i in range(self.NBmW):
                for j in range (self.NBdW):
                    for m in range (self.NP):
                        self.CIR = (20/self.BmW_b[i]) - self.P_b[m]
                        #print(CIR)
                        self.G = 0.7*((4*3.1416)/((self.BmW_b[i]*(3.1416/180))**(2)))
                        self.PIRE = self.P_b[m] + (10*math.log10(self.G))
                        #print(PIRE)
                        self.CNR = 228.6 + self.PIRE + self.G_T - self.Loss - 10*math.log10(self.BdW_b[j]*(10**6))
                        #print(CNR)
                        self.CINR = 10*math.log10(1/((1/(10**(self.CNR/10))) + (1/(10**(self.CIR/10)))))
                        #print(CINR)
                        self.SE = (0.1851*self.CINR) + 0.6905 #eficiencia espectral
                        self.Cap = (self.SE*(self.BdW_b[j]*(10**6)))/(10**9)
                        self.C_b[i, j, m] = self.Cap
                        #print(C_b)
            self.X, self.Y = np.meshgrid(self.P_b,self.BdW_b)
            State_space1 = np.reshape(self.C_b[0,:,:], -1)
            State_space2 = np.reshape(self.C_b[1,:,:], -1)
            State_space3 = np.reshape(self.C_b[2,:,:], -1)
            self.State_space = np.concatenate((State_space1, State_space2, State_space3))
            return self.C_b, self.X, self.Y, self.State_space
    
        def Traffic_b(self):
            self.hora = np.linspace(1, 24, num=24)
            self.R_b = np.array([0.582730536227721, 0.358208163494681, 0.166169147345066, 0.130264713001518, 0.0660307931380860, 1.07042345707438, 1.56283610631388, 2.22224701521823, 3.09905819551566, 3.19322070371709, 3.07415523419705, 3.34902787311339, 2.78229807926008, 2.46027767231635, 2.83970730739323, 3.21583625350833, 2.91960152214340, 3.35789194025091, 3.12768001691867, 3.06319092722659, 3.62610476514055, 3.29821063120155, 2.51833296631518, 1.49190780349331]) 
            self.R_b = 0.4 * self.R_b
            return self.R_b, self.hora
            #ax4 = plt.axes(projection='2d') 
            
        def _take_action(self, action):
            #t = random.randint(1,24)
            #self.observation_space(t)
            action_type = action
            #if action_type == 0:
             #   self.agentPositionP = self.agentPositionP
              #  self.agentPositionBdW = self.agentPositionBdW 
              #  self.agentPositionBmW = self.agentPositionBmW 
            if action_type == 1 and self.agentPositionP < self.NP - 1:
                self.agentPositionP = self.agentPositionP + 1
            
            elif action_type == -1 and self.agentPositionP > 0:
                self.agentPositionP = self.agentPositionP - 1
            
            elif action_type == -2 and self.agentPositionBdW > 0:
                self.agentPositionBdW = self.agentPositionBdW - 1
            
            elif action_type == 2 and self.agentPositionBdW < self.NBdW - 1:
                self.agentPositionBdW = self.agentPositionBdW + 1
            
            elif action_type == -3 and self.agentPositionBmW > 0:
                self.agentPositionBmW = self.agentPositionBmW - 1
            
            elif action_type == 3 and self.agentPositionBmW > self.NBmW - 1:
                self.agentPositionBmW = self.agentPositionBmW + 1
            
            else:
                self.agentPositionP = self.agentPositionP
                self.agentPositionBdW = self.agentPositionBdW 
                self.agentPositionBmW = self.agentPositionBmW 
            
        def step(self, action):
            #old_ob = self.observation_space(t)
            state = self.space()
            self.R_b_t = R_b_t
            old_C_b_t = C_b_t
            self._take_action(action)
            state = self.space()
            new_C_b_t = C_b_t
            self.C_b_t = new_C_b_t
            self.state = state
            reward = - 0.05*(abs(new_C_b_t - R_b_t))
            return self.observation, reward[0], self.isTerminalState(self.observation), None
        
    
        def space(self):
            self.Traffic_b()
            self.R_b_t = self.R_b[self.t-1]
            #self.CbSpace()
            self.C_b_t = self.C_b[self.agentPositionBmW, self.agentPositionBdW, self.agentPositionP]
            #self.P_b_t = self.P_b[self.agentPositionP]
            #self.BmW_b_t = self.BmW_b[self.agentPositionBmW]
            #self.BdW_b_t = self.BdW_b[self.agentPositionBdW]
            self.state1 = np.where(self.observation_space == self.C_b_t)
            self.observation = self.state1[0]
            #print(self.State_space)
            #print(self.agentPositionBmW)
            #print(self.agentPositionBdW)
            #print(self.agentPositionP)
            #print(self.C_b_t)
            #print(int(self.state))
            #print(self.state)
        
            return self.R_b_t, self.C_b_t, self.observation
        
        def reset(self):
            self.P_b_t = 15
            self.BmW_b_t = 100
            self.BdW_b_t = 0.6
            self.agentPositionP1 = np.where(self.P_b == self.P_b_t)
            self.agentPositionP = self.agentPositionP1[0]
            self.agentPositionBdW1 = np.where(self.BdW_b == self.BdW_b_t) 
            self.agentPositionBdW = self.agentPositionBdW1[0]
            self.agentPositionBm1 = np.where(self.BmW_b == self.BmW_b_t)
            self.agentPositionBmW = self.agentPositionBm1[0]
            self.CbSpace()
            self.space()
            return self.observation
        
        def render(self):
            print (f'Required Traffic: {self.R_b_t}')
            print (f'Offered Traffic: {self.C_b_t}')
       
        def actionSpaceSample(self):
            return np.random.choice(self.possibleActions)
