# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 01:10:06 2021

@author: xupeng
"""



import numpy as np
import PengMethodList as pm


class AllStates(object):
    
    def __init__(self, **kw):   
        self.BN = kw.get('BN')  
        self.VSN = self.BN.VSN
        self.VEN = self.BN.VEN
        
        self.StateCount = 2*pow(2,(self.VSN-1))*pow(3,(self.VEN-self.VSN+1))
    
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
        for i in range(NumOfVA):
            Rem = Index_2 - int(Index_2/3)*3
            if Rem == 1:
                Key[-i-1] = -1
            if Rem == 2:
                Key[-i-1] = 1
                
            Index_2 = int(Index_2/3)
                
        for i in range((self.VSN-1)):
            Rem = Index_2 - int(Index_2/2)*2
            Key[-i-1-NumOfVA] = Rem
            
            Index_2 = int(Index_2/2)
            
        CAOrNot = Index_2
        
        return Key, CAOrNot
    
    
    def changeKey2Index(self, Key, CAOrNot):
        NumOfVA = self.VEN-self.VSN+1
        Index = 0
        
        for i in range(NumOfVA):
            Rem = 0
            if Key[-i-1] == -1:
                Rem = 1
            if Key[-i-1] == 1:
                Rem = 2
            Index = Index + Rem*pow(3,i)
            
        for i in range((self.VSN-1)):
            Rem = Key[-i-1-NumOfVA]
            Index = Index + Rem*pow(2,i)*pow(3,NumOfVA)
            
        if CAOrNot == 1:
            Index = Index + pow(2,(self.VSN-1))*pow(3,NumOfVA)
            
        return Index
    
    
    def updateOneStateOrder(self,Index):
        Key,CAOrNot =  self.changeIndex2Key(Index)
        Current_Order = self.AllStateOrders[Index,0]
        
        AllCAState_List = Key[0:(self.VSN-1)]
        AllVAState_List = Key[(self.VSN-1):self.VEN]
        
        
        Updated = 0
    
        if CAOrNot == 0:
            ActionList = [(i+(self.VSN-1)) for i, e in enumerate(AllVAState_List) if e == 0]
            ActionList = ActionList            # dont need the action (no VA: -1) as the first action
            ActionCount = len(ActionList)
            
            
            # debug
            #if Index == 1:
            #    print('ActionList:',ActionList)
                
            
            for i in range(ActionCount):
                Action = ActionList[i]          
                    
                # get the next node keys
                NumOfNextState, NextNodeKeys, Next_CAOrNot = self.getNextStateKeys(CAOrNot,Key,Action) # NextNodeKeys is a LIST
                
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
                        
    
        # current activity is CA and the result of last VA is false            
        elif CAOrNot == 1:
            ActionList = [i for i, e in enumerate(AllCAState_List) if e == 0]
            ActionList = [-1]+ActionList            # plus the action (no VA: -1) as the first action!
            ActionCount = len(ActionList)
            
            #CandidateValueList = np.zeros((2,ActionCount)) 
            
            ValidOrNot = 1
            AllVAState_List = Key[(self.VSN-1):self.VEN]
            if CAOrNot == 1 and sum(map(abs,AllVAState_List)) == 0:
                ValidOrNot = 0
                
            if ValidOrNot == 1:
                for i in range(ActionCount):
                    Action = ActionList[i]
                    
                    # get the next node keys
                    NumOfNextState, NextNodeKeys,Next_CAOrNot = self.getNextStateKeys(CAOrNot,Key,Action) # NextNodeKeys is a LIST
                    
                    if NumOfNextState == 1:                
                        Next_Index = self.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
                        Next_Order = self.AllStateOrders[Next_Index,0]
    
                        if Next_Order <= Current_Order:
                            self.AllStateOrders[Next_Index,0] = Current_Order + 1
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
                Next_CAOrNot = [1]
                
                #print('Error 1 of Action!')
                #print(Action)
                
            elif Action >= (self.VSN-1):
                NumOfNextState = 2
                Key_1 = Key.copy()
                Key_1[Action] = 1
                Key_2 = Key.copy()
                Key_2[Action] = -1
                
                NextNodeKeys = [Key_1, Key_2]
                # if current is VA and its result is true,the next CA state is skipped and go to the next VA instead.
                Next_CAOrNot = [1,1]
            
            else:
                print('Error 2 of Action!')
                print(Action)
            
        # current activity is CA and the result of last VA is true  
        #if NodeType == 1:
        #    NumOfNextState = 1            
        #    Key = NodeKey.copy()
        #    NextNodeKeys = [Key]
                
        # current activity is CA and the result of last VA is false     
        if CAOrNot == 1:
            Key_1 = Key.copy()
            if Action == -1  or Action == -2:    # no CA
                NumOfNextState = 1    
                NextNodeKeys = [Key_1]
                Next_CAOrNot = [0]
            elif Action >= 0 and Action < (self.VSN-1):   # CA   
                ChildList = self.BN.findAllChildNode(Action)
                for i in ChildList:
                    if i >= (self.VSN-1):
                        Key_1[i] = 0
                    
                Key_1[Action] = 1                
                NumOfNextState = 1    
                NextNodeKeys = [Key_1]    # need update!
                Next_CAOrNot = [0]
                    
            else:
                print('Error of Action!')
                print(Action)    
                
        return NumOfNextState, NextNodeKeys,Next_CAOrNot
        
    
    def saveAllStates(self, FileName):
        Dict = {'AllStateValue' : self.AllStateValues, 'VACost': self.BN.VACost, 'CACost': self.BN.CACost\
            , 'AllStateOrders': self.AllStateOrders, 'AllStateActions': self.AllStateActions, 'AllStateLabels': self.AllStateLabels\
            , 'FailCost': self.BN.FailCost, 'RepairLRVector': self.BN.RepairLRVector}

        # save
        pm.saveDictVariables(Dict,FileName)