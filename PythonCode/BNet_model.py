# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 18:08:42 2019

@author: xupeng
"""


import numpy as np

#import sys
#if sys.version_info.major == 2:
#    import Tkinter as tk
#else:
#    import tkinter as tk
    
    
import matlab
import matlab.engine


#UNIT = 40   # pixels
#MAZE_H = 4  # grid height
#MAZE_W = 4  # grid width


class BNet(object):
    def __init__(self):
        super(BNet, self).__init__()
        
        #self.action_space = ['u', 'd', 'l', 'r']
        #self.n_actions = len(self.action_space)
        
        self.engine = matlab.engine.start_matlab()
        #self.Evidence = [0]*12
        
        # the BN model should be the same as the system
        # use the BN structure in the belief fusion paper but VSN starts at 6
        self.VSN = 7
        self.VEN = 12
        self.N = self.VEN + self.VSN - 1
        self.TargetNode = 5         # mean the last node  6-1
        self.LevelNum = 2
        #self.TimeLimit = 5
        #self.PenaltyCoeff = [1, 1.1067, 1.2247, 1.3554, 1.5]
        
        self.NodeTable_np = np.zeros((self.N,self.N))
        self.FactorMatrix_np = np.zeros((self.N,self.N))
        self.NodeTable_matlab = None     
        self.FactorMatrix_matlab = None
        
        
        #self.ReworkLRVector = np.zeros((1,(self.VSN-1)))   
        # prob. that reworked element is true
        self.ReworkLRVector = np.array([[0.85, 0.85, 0.85, 0.83, 0.6, 0.9]])
                
        
        #self.RepairLRVector = np.array([[18, 19, 11.875, 18, 7.5]]) # for repair
        self.RepairLRVector = np.array([[18, 18, 18, 8, 3.5, 8]]) # for repair     need check!
                
        #self.TempRevenue = TempRevenue
        #self.TempActCost = TempActCost
        #self.TempReworkCost = TempReworkCost
        
        
        
        self.Coeff1_forMC = 1000   # between cost and NN output/MC selection
        
        self.Revenue = 20000
        self.VACost =  [0,0,0,0,0,0,200,200,200,400,100,500]    # VEN - VSN         
        #self.FailCost = [0,0,0,0,0,0,1000,1000,1000,2000,0,12000]
        #self.CACost = [1000, 1000, 1000,2000,500,7000,0,0,0,0,0,0]  # VSN     [19000, 8460, 12030,4730,3820]
        #self.CACost4ThresRule = [0,0,0,0,0,0, 1000,1000,1000,2000,500,7000]  # VSN     [19000, 8460, 12030,4730,3820]
        
        self.FailCost = [0,0,0,0,0,0,1000,1000,1000,3000,0,15000]
        self.CACost = [1000, 1000, 1000,2000,500,8000,0,0,0,0,0,0]  # VSN     [19000, 8460, 12030,4730,3820]
        self.CACost4ThresRule = [0,0,0,0,0,0, 1000,1000,1000,2000,500,8000]  # VSN     [19000, 8460, 12030,4730,3820]
        
        
        #self.Revenue = self.Revenue/self.Coeff1_forMC
        #self.VACost = [i/self.Coeff1_forMC for i in self.VACost]
        #self.CACost = [i/self.Coeff1_forMC for i in self.CACost]
        
        
        self.UpperThres = 0.90
        self.LowerThres = 0.20
        #self.LowerThres = [0.2, 0.3, 0.575, 0.85, 0.95]
        #self.LowerThres = [0.95, 0.95, 0.95, 0.95, 0.95]

        self.inputnum = self.N  # confidence of all parameters and node states
        self.outputnum = 1+1
        
        self._load_BNT()
        self._init_2Matrices()
        self._build_BNet()
        
    def _load_BNT(self):        
        #self.engine.cd(r'C:\Users\xupen\Google Drive\PythonScripts\BayesNet\MatlabCode', nargout=0)
        self.engine.cd(r'D:\GoogleDrive\PythonScripts\JVS\MatlabCode', nargout=0)        
        self.engine.loadBNT(nargout = 0)

    def _init_2Matrices(self):
                
        self.NodeTable_np[0:self.VEN,0:self.VEN] = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                                                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        for i in range(self.VSN-1):
            self.NodeTable_np[i,i+self.VEN] = 1            
        
        self.NodeTable_matlab = matlab.double(self.NodeTable_np.tolist())    
        
         
        # for parameter:    when X is false, what is the prob. that Y is false?
        #                   diagonal： the prob. that Y is false by itself
        # for measurement:   value in Row 1: when X is false, the prob. that Y is false.
        #                    value on the diagonal: when X is true, the prob. that Y is true.
                
        # 0713-2
        self.FactorMatrix_np[0:self.VEN,0:self.VEN] = np.array([[0.08, 0.00, 0.00, 0.80, 0.00, 0.00, 0.95, 0.95, 0.95, 0.90, 0.80, 0.90],
                                                                [0.00, 0.08, 0.00, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.08, 0.00, 0.00, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.08, 0.00, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.08, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.00, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.00],
                                                                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80]])
        
    
        for i in range(self.VSN-1):
            self.FactorMatrix_np[(i+self.VEN),(i+self.VEN)] = self.RepairLRVector[0,i]
        
        self.FactorMatrix_matlab = matlab.double(self.FactorMatrix_np.tolist())  
        
        
    def _build_BNet(self):
        self.engine.cd(r'D:\GoogleDrive\PythonScripts\JVS\MatlabCode', nargout=0)        
        self.engine.BNet_generate_4JVP(self.VSN,self.VEN,self.LevelNum,self.NodeTable_matlab,self.FactorMatrix_matlab, nargout=0)
        
        
    def inferBNet(self, Evidence):        
        #self.engine.cd(r'C:\Users\xupen\Google Drive\PythonScripts\BayesNet\MatlabCode', nargout=0)
        self.engine.cd(r'D:\GoogleDrive\PythonScripts\JVS\MatlabCode', nargout=0)
        
        #print(Evidence)
        InitPriorVector = self.engine.inferenceBN(Evidence)
        #print(InitPriorVector)

        PyPriorVector = 1 - np.asarray(InitPriorVector).flatten() # prob of true

        #self.engine.exit()        
        return PyPriorVector
    
    
    def changeState2Evidence(self,NodeKey):
        Evidence = self.resetEvidence()        
        for i in range(self.VSN-1,self.VEN):
            if NodeKey[i] == 1:
                Evidence[i] = 2
            if NodeKey[i] == -1:
                Evidence[i] = 1
                
        return Evidence
        
# =============================================================================
#     def inferWithSystemStateAfterRework(self, NodeKey):
#         ReworkedOrNot = 0
#         
#         TempFactorMatrix = self.FactorMatrix_np.copy()  # bug !!!
#         
#         for i in range(self.VSN-1):
#             if NodeKey[i] == 1:
#                 
#                 ReworkedOrNot = 1
#                 
#                 ReworkNode = i
#                 # update cause factor
#                 OriginalProb = TempFactorMatrix[ReworkNode,ReworkNode]            
#                 LR = self.ReworkLRVector[0,ReworkNode]            
#                 NewProb = (1-OriginalProb)*LR/(OriginalProb*(1-LR)+(1-OriginalProb)*LR)     # prob. of true
#                 
#                 TempFactorMatrix[ReworkNode,ReworkNode] = 1 - NewProb
#                     
#         if ReworkedOrNot == 1:
#             self.FactorMatrix_matlab = matlab.double(TempFactorMatrix.tolist())  
#             
#             self._build_BNet();
#         
#         Evidence = self.changeState2Evidence(NodeKey)
# 
#         PyPriorVector = self.inferBNet(Evidence)       # prob of true result
#         
#         return PyPriorVector
# =============================================================================
    
    
    def inferWithSystemStateAfterRepair(self, NodeKey):
        #RepairedOrNot = 0
        Evidence = self.changeState2Evidence(NodeKey)
        
        #TempFactorMatrix = self.FactorMatrix_np.copy()  # bug !!!
        
        for i in range(self.VSN-1):
            if NodeKey[i] == 1:
                
        #        RepairedOrNot = 1                
                
                # update cause factor         
        #        LR = self.RepairLRVector[0,i]                            
        #        TempFactorMatrix[(i+self.VEN),(i+self.VEN)] = LR
                
                # update evidence
                Evidence[(i+self.VEN)] = 2  
                
        #if RepairedOrNot == 1:
        #    self.FactorMatrix_matlab = matlab.double(TempFactorMatrix.tolist())  
        #    self._build_BNet();
            
            #print(Evidence)
            #print(TempFactorMatrix)
        
        PyPriorVector = self.inferBNet(Evidence)       # prob of true result
        
        return PyPriorVector
    
    
    
    def getVACost(self,ActionNum):
        if ActionNum >= (self.VSN-1) and ActionNum <= (self.VEN-1):
            VACost = self.VACost[ActionNum]
        elif ActionNum == -1:
            VACost = 0
        else:
            print('Error of VA index!')
            print(ActionNum)
        return VACost
        
    def getFailCost(self,ActionNum):
        if ActionNum >= (self.VSN-1) and ActionNum <= (self.VEN-1):
            FailCost = self.FailCost[ActionNum]
        #elif ActionNum == -1:
        #    FailCost = 0
        else:
            print('Error of VA index!')
            print(ActionNum)
        return FailCost
        
    def getCACost(self,ActionNum):
        if ActionNum >= 0 and ActionNum < (self.VSN-1):
            CACost = self.CACost[ActionNum]
        elif ActionNum == -1:
            CACost = 0
        else:
            print('Error of CA index!')
            
        return CACost
    
    def getCACost4ThresRule(self,ActionNum):
        if ActionNum >= (self.VSN-1) and ActionNum <= (self.VEN-1):
            CACost = self.CACost4ThresRule[ActionNum]
        elif ActionNum == -1:
            CACost = 0
        else:
            print('Error of CA index!')
            print(ActionNum)
            
        return CACost

    def resetEvidence(self):
        # the length is self.N
        # the first VSN nodes represent CA: 0: not reworked yet; 1: reworked
        # the next (VEN-VSN) nodes represent VA: 0: not verified; -1: the VA result is false; 1: the VA result is true.
        # the rest nodes are not used temporarily.
        
        self.Evidence = [0]*self.N
        return self.Evidence
    
    
    def checkDoneOrNot(self, NodeKey): 
        # each node can be conducted once. If all nodes are conducted, it is done.
        done = False
        if sum(abs(np.array(NodeKey[0:self.VEN]))) == self.VEN:
            done = True            
        
        ConfidenceVector = self.inferWithSystemStateAfterRepair(NodeKey)
        PostProb = ConfidenceVector[self.TargetNode]
        if  PostProb > self.UpperThres:
            done = True
            StateValue = self.Revenue*PostProb
        else:
            StateValue = 0
        
        ParaConfidence = ConfidenceVector[0:(self.VSN-1)]
            
        return ParaConfidence, done, StateValue
    
    
    def obtainTestNodeProb(self, NodeKey, TestNode):
        # TestNode: start from 0, not 1
        
        PyPriorVector = self.inferWithSystemStateAfterRepair(NodeKey)
        PostProb = PyPriorVector[TestNode] # prob for true        
        
        return PostProb


    def findAllChildNode(self,ParentNode):
        # range is a list
        Range = list(range(self.VEN))
        
        CandidateList = [ParentNode]
        UpdateList = [ParentNode]
        
        while len(CandidateList) > 0:
            CurrentNode = CandidateList[0]
            del CandidateList[0]
            
            # find nearest child nodes
            TempRow = self.NodeTable_np[CurrentNode,Range]
            temp = np.argwhere(TempRow == 1)
            
            for i in range(len(temp)):
                if temp[i][0] not in UpdateList:
                    CandidateList.append(temp[i][0])
                    UpdateList.append(temp[i][0])
        
        return UpdateList

    #def updateEvidenceAfterCA(self,ReworkNode, Evidence):
    #    if ReworkNode >= (self.VSN-1):
    #        print('Error in selection of rework node')
    #    
    #    else:
    #        UpdateList = self.findAllChildNode(ReworkNode)
    #        
    #        for i in UpdateList:
    #            if i >= (self.VSN-1):
    #                Evidence[i] = 0
    #                
    #    return Evidence
        
        
if __name__ == '__main__':
    env = BNet()
    #env.after(100, update)
    #env.mainloop()