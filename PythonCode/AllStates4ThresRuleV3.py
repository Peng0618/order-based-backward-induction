# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 13:00:19 2021

@author: xupeng
"""


import numpy as np

from BNet_model import BNet

class AllStates4ThresRule(object):
    
    def __init__(self, **kw):   
        self.BN = kw.get('BN')  
        self.VSN = self.BN.VSN
        self.VEN = self.BN.VEN
        
        self.StateCount = pow(4,(self.VEN-self.VSN+1))
    
        self.AllStateValues = np.full((self.StateCount,1),-20000.0)
        
        # label: (0: not updated; 1: updated; 2: ended) 
        self.AllStateLabels = np.zeros((self.StateCount,1))
                
        self.AllStateOrders = np.zeros((self.StateCount,1))
            
        self.AllStateActions = np.full((self.StateCount,1), -10)
        
        #self._initilizeWithInitNode()
        
        
    def changeIndex2Key(self, Index):
        NumOfVA = self.VEN-self.VSN+1
        CAOrNot = 0 # 0: VA; 1: CA and previous VA is false
        Key = [0]*self.VEN
        Index_2 = Index
        
        Digit= 4
        for i in range(NumOfVA):
            Rem = Index_2 - int(Index_2/Digit)*Digit
            if Rem == 1:
                Key[-i-1] = -1
            if Rem == 2:
                Key[-i-1] = 1
            if Rem == 3:
                Key[-i-1-NumOfVA] = 1
                            
            Index_2 = int(Index_2/Digit)
                
        #for i in range((self.VSN-1)):
        #    Rem = Index_2 - int(Index_2/Digit)*Digit
        #    Key[-i-1-NumOfVA] = Rem
        #    
        #    Index_2 = int(Index_2/2)
            
        CAOrNot = Index_2
        
        return Key, CAOrNot
    
    
    def changeKey2Index(self, Key, CAOrNot):
        NumOfVA = self.VEN-self.VSN+1
        Index = 0
        
        Digit= 4
        for i in range(NumOfVA):
            Rem = 0
            if Key[-i-1] == -1:
                Rem = 1
            if Key[-i-1] == 1:
                Rem = 2
            if Key[-i-1-NumOfVA] == 1:
                Rem = 3
                
            Index = Index + Rem*pow(Digit,i)
            
        #for i in range((self.VSN-1)):
        #    Rem = Key[-i-1-NumOfVA]
        #    Index = Index + Rem*pow(2,i)*pow(Digit,NumOfVA)
            
        if CAOrNot == 1:
            print('Error in changeKey2Index')
        #    Index = Index + pow(2,(self.VSN-1))*pow(Digit,NumOfVA)
            
        return Index
    
    
    def updateOneStateOrder(self,Index):
        Key,CAOrNot =  self.changeIndex2Key(Index)
        Current_Order = self.AllStateOrders[Index,0]
        
        #AllCAState_List = Key[0:(self.VSN-1)]
        #AllVAState_List = Key[(self.VSN-1):self.VEN]
        
        
        Updated = 0
    
        if CAOrNot == 0:
            
            TestKey = self.getTestKey(Key)            
            if max(TestKey) == 2:
                ActionList = []
                self.AllStateOrders[Index,0] = -10 # special number for illegal states.
                
            else:                                
                ActionList = [(i+(self.VSN-1)) for i, e in enumerate(TestKey) if e == 0]
                ActionList = ActionList            # dont need the action (no VA: -1) as the first action
                        
            ActionCount = len(ActionList)
            
            
            # debug
            #if Index == 1:
            #    print('ActionList:',ActionList)
                
            
            for i in range(ActionCount):
                Action = ActionList[i]          
                    
                # get the next node keys
                NumOfNextState, NextNodeKeys, Next_CAOrNot, ReworkOrNot = self.getNextStateKeys(CAOrNot,Key,Action) # NextNodeKeys is a LIST
                
                if NumOfNextState == 2:      
                    Next_Index_1 = self.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
                    Next_Order_1 = self.AllStateOrders[Next_Index_1,0]
                    
                    Next_Index_2 = self.changeKey2Index(NextNodeKeys[1], Next_CAOrNot[1])
                    Next_Order_2 = self.AllStateOrders[Next_Index_2,0]
                    
                    if Next_Order_1 <= Current_Order:
                        self.AllStateOrders[Next_Index_1,0] = Current_Order + 1
                        Updated = 1
                        
                    if Next_Order_2 <= Current_Order:
                        self.AllStateOrders[Next_Index_2,0] = Current_Order + 1
                        Updated = 1
                        
                
        return Updated
        
        
    # for state update
    def getNextStateKeys(self, CAOrNot, Key, Action):     
        # node is state
        # current activity is VA
        if CAOrNot == 0:
            if Action == -1 or Action == -2:
                Key_1 = Key.copy()
                NumOfNextState = 0
                NextNodeKeys = [Key_1]
                Next_CAOrNot = [0]
                ReworkOrNot = [0]
                
                #print('Error 1 of Action!')
                #print(Action)
                
            elif Action >= (self.VSN-1):
                NumOfNextState = 2
                Key_1 = Key.copy()
                Key_1[Action] = 1
                
                Key_2 = Key.copy()
                Key_2[Action] = -1
                
                PostProb = self.BN.obtainTestNodeProb(Key_2, self.BN.TargetNode)
                
                if PostProb <= self.BN.LowerThres:        
                    # modified at 0727:
                    CorrectionNum = Action-(self.VSN-1)    # need check!!!
                    
                    ChildList = self.BN.findAllChildNode(CorrectionNum)
                    for i in ChildList:
                        if i >= (self.BN.VSN-1):
                            Key_2[i] = 0
                        
                    Key_2[CorrectionNum] = 1
                    ReworkOrNot = [0,1]
                                        
                else:
                    ReworkOrNot = [0,0]
                
                NextNodeKeys = [Key_1, Key_2]
                # if current is VA and its result is true,the next CA state is skipped and go to the next VA instead.
                Next_CAOrNot = [0,0]
            
            else:
                print('Error 2 of Action!')
                print(Action)
            
                
        return NumOfNextState, NextNodeKeys, Next_CAOrNot, ReworkOrNot
    
    
    # for the calculation of negative result case
    def getFalseResultKey(self, CAOrNot, Key, Action):     
        # node is state
        # current activity is VA
        if CAOrNot == 0:
            if Action == -1 or Action == -2:
                Key_2 = Key.copy()
                
            elif Action >= (self.VSN-1):
                Key_2 = Key.copy()
                Key_2[Action] = -1
                
            
            else:
                print('Error 2 of Action!')
                print(Action)
            
                
        return Key_2
    
    def getTestKey(self,Key):
        Temp = abs(np.array(Key))
        Temp2 = Temp[0:(self.VSN-1)]+Temp[(self.VSN-1):self.VEN]
        TestKey = Temp2.tolist()
        
        return TestKey
    
    
if __name__ == "__main__":
    
    # BN model
    BN = BNet()        
    
    AllStateSet4ThresRule = AllStates4ThresRule(BN = BN)