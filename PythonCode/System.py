# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 01:29:04 2021

@author: xupeng
"""

# this class is mainly used to generate samples about system elements

import numpy as np


#UNIT = 40   # pixels
#MAZE_H = 4  # grid height
#MAZE_W = 4  # grid width


class System(object):
    def __init__(self):        
        super(System, self).__init__()        
        
        # use the BN structure in the belief fusion paper but VSN starts at 6
        self.VSN = 6
        self.VEN = 14
        self.N = self.VEN + self.VSN - 1
        self.TargetNode = 0         # mean the first node  1-1
        self.LevelNum = 2
        #self.TimeLimit = 5
        #self.PenaltyCoeff = [1, 1.1067, 1.2247, 1.3554, 1.5]
        
        
        self.NodeTable_np = np.zeros((self.N,self.N))
        self.FactorMatrix_np = np.zeros((self.N,self.N))
                
        self.ReworkLRVector = np.zeros((1,(self.VSN-1)))       
                
        self.OneSampleCauseFactors = np.zeros(((self.VSN-1),(self.VSN-1)))
        self.OneSampleParameters = np.zeros((1,(self.VSN-1)))
                        
        self._init_2Matrices()
        
    def _init_2Matrices(self):
                
        self.NodeTable_np[0:self.VEN,0:self.VEN] = np.array([   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                                                [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                                                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                                                          
        for i in range(self.VSN-1):
            self.NodeTable_np[i,i+self.VEN] = 1         
             
        
         
        # for parameter: when X is false, what is the prob. that Y is false?
        # for measurement:   value in Row 1: when X is false, the prob. that Y is false.
        #                    value on the diagonal: when X is true, the prob. that Y is true.
        self.FactorMatrix_np[0:self.VEN,0:self.VEN] = np.array([[0.30, 0.00, 0.00, 0.00, 0.00, 0.95, 0.85, 0.95, 0.80, 0.92, 0.88, 0.78, 0.95, 0.88],
                                                             [0.25, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.15, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.30, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.20, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.95, 0.00, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.00, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.00, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.90, 0.00],
                                                             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.90]])
        
        for i in range(self.VSN-1):
            self.FactorMatrix_np[(i+self.VEN),(i+self.VEN)] = 1        
            
        # prob. that reworked element is true
        self.ReworkLRVector = np.array([[0.7, 0.82, 0.86, 0.83, 0.9]])
        
        

    
    def generateOneSystemSample(self):            
        
        # generate a set of sample cause factors:
        for i in range(self.VSN-1):         # for each parameter
            for j in range(self.VSN-1):     # for each parent parameter
                # 2 success; 1: error; 0: not exist
                if i != j and self.NodeTable_np[j,i] == 0:
                    self.OneSampleCauseFactors[j,i] = 0                
                else:
                    p = 1 - self.FactorMatrix_np[j,i]   # probability of success
                
                    CauseFactorValue = np.random.binomial(1,p)  
                    self.OneSampleCauseFactors[j,i] = CauseFactorValue + 1;
                
        # generate a sampling sequence of all system parameters:        
        ParaSeq = np.arange(0, (self.VSN-1))  
        for i in range(self.VSN-1):         # for each parameter
            for j in range(self.VSN-1):     # for each parent parameter
                if self.NodeTable_np[j,i] == 1:
                    Index1 = np.argwhere(ParaSeq == i)[0][0]
                    Index2 = np.argwhere(ParaSeq == j)[0][0]
                    
                    if Index1 < Index2:
                        ParaSeq[Index1], ParaSeq[Index2] = ParaSeq[Index2], ParaSeq[Index1];
                    
        # generate a sample of all parameters: 2 success; 1: error; 0: not exist
        for i in ParaSeq:                   # for each parameter
            self.OneSampleParameters[0][i] = 2;
            for j in range(self.VSN-1):     # for each parent parameter
                if i == j:
                    if self.OneSampleCauseFactors[j,i] == 1:
                        self.OneSampleParameters[0][i] = 1;
                        
                elif self.OneSampleCauseFactors[j,i] == 1:  # there is a false connection
                    if self.OneSampleParameters[0][j] == 1:
                        self.OneSampleParameters[0][i] = 1;
                    if self.OneSampleParameters[0][j] == -1:
                        print('Error in sample generation')    
        
    
    def getOneMeasurement(self, TestNode, Evidence):    
        if TestNode < self.VSN - 1 or TestNode >= self.VEN:
            print('Error in selection of testing node')
            
        else:
            # find parent test nodes
            ColVector = self.NodeTable_np[0:self.VEN,TestNode]
            ParentList = np.argwhere(ColVector == 1)
            ParentNum = ParentList.size
            if ParentNum < 1 or ParentNum > 2:
                print('Error in parent node number')
                            
            
            # determine distribution
            if ParentNum == 1:
                # get parnet node state
                ParameterIndex = ParentList[0][0]
                ParameterValue = self.OneSampleParameters[0][ParameterIndex]
                if ParameterValue == 2:
                    SamplingDist = self.FactorMatrix_np[TestNode,TestNode]  # true prob.
                else:
                    SamplingDist = 1 - self.FactorMatrix_np[0,TestNode]         # true prob.
                        
            if ParentNum == 2:
                ParameterIndex = ParentList[0][0]                
                ParameterValue = self.OneSampleParameters[0][ParameterIndex]
                AnotherTestNodeIndex = ParentList[1][0]
                
                if Evidence[AnotherTestNodeIndex] > 0:
                    if Evidence[AnotherTestNodeIndex] == 2 and ParameterValue == 2:
                        SamplingDist = self.FactorMatrix_np[TestNode,TestNode]  # true prob.
                    elif Evidence[AnotherTestNodeIndex] == 1 and ParameterValue == 1:
                        SamplingDist = 1 - self.FactorMatrix_np[0,TestNode]     # true prob.
                    else:
                        SamplingDist = 0.5;
                elif Evidence[AnotherTestNodeIndex] == 0:
                    # calc the marginal dist. of the test node
                    ANTT = self.FactorMatrix_np[AnotherTestNodeIndex,AnotherTestNodeIndex]
                    ANFF = self.FactorMatrix_np[0,AnotherTestNodeIndex]
                    #AnotherTestNodeDist = np.array([[ANFF, 1-ANTT, 1-ANFF, ANTT]])
                    TNTT = self.FactorMatrix_np[TestNode,TestNode]
                    TNFF = self.FactorMatrix_np[0,TestNode]
                    
                    MarginalDist_TT = ANTT*TNTT + (1-ANTT)*0.5
                    MarginalDist_FF = ANFF*TNFF + (1-ANFF)*0.5
                    
                    if ParameterValue == 2:
                        SamplingDist = MarginalDist_TT;
                    elif ParameterValue == 1:
                        SamplingDist = 1 - MarginalDist_FF;
                               
                
            # get a sample measurement
            MeasurementResult = np.random.binomial(1,SamplingDist) + 1 
            Evidence[TestNode] = MeasurementResult
            
            return Evidence
        
    
    
    def applyReworkToOneNode(self, ReworkNode, Evidence):
        if ReworkNode >= (self.VSN-1):
            print('Error in selection of rework node')
        
        else:
            # resample cause factor
            OriginalProb = self.FactorMatrix_np[ReworkNode,ReworkNode]            
            LR = self.ReworkLRVector[0,ReworkNode]            
            NewProb = (1-OriginalProb)*LR/(OriginalProb*(1-LR)+(1-OriginalProb)*LR)     # prob. of true
            
            NewCauseFactorValue = np.random.binomial(1,NewProb)  
            self.OneSampleCauseFactors[ReworkNode,ReworkNode] = NewCauseFactorValue + 1;
            
            # update parameters
            ParaRange = list(range(self.VSN-1))
            UpdateParaList = self.findAllChildNode(ReworkNode,ParaRange)
            
            for i in UpdateParaList:
                self.OneSampleParameters[0][i] = 2;
                for j in range(self.VSN-1):     # for each parent parameter
                    if i == j:
                        if self.OneSampleCauseFactors[j,i] == 1:
                            self.OneSampleParameters[0][i] = 1;
                        
                    elif self.OneSampleCauseFactors[j,i] == 1:  # there is a false connection
                        if self.OneSampleParameters[0][j] == 1:
                            self.OneSampleParameters[0][i] = 1;
                        if self.OneSampleParameters[0][j] == -1:
                            print('Error in sample generation')
            
            # update evidence
            AllNodeRange = list(range(self.VEN))            
            UpdateList = self.findAllChildNode(ReworkNode,AllNodeRange)
            
            for i in UpdateList:
                if i >= (self.VSN-1):
                    Evidence[i] = 0
                    
            return Evidence
            
            
    def findAllChildNode(self,ParentNode,Range):
        # range is a list
        
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
    
if __name__ == '__main__':
    sys = System()
    sys.generateOneSystemSample()
    
    Evidence = sys.resetEvidence()    
    TestNode = 5
    Evidence = sys.getOneMeasurement(TestNode,Evidence)
    
    # sys.OneSampleParameters
    # sys.OneSampleCauseFactors
    
    
    ReworkNode = 0
    Evidence = sys.applyReworkToOneNode(ReworkNode, Evidence)
    

    
    #env.after(100, update)
    #env.mainloop()        