# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:38:41 2021

@author: xp

Value iteration

"""


import numpy as np

#from System import System
from BNet_small_model import BNet

#import copy
import time

import PengMethodList as pm



def update_ValueFunction(BN,AllStateRecord):
    StateCount = len(AllStateRecord)
    MaxError = 0
    for i in range(StateCount):
        Key,CAOrNot =  changeIndex2Key(i,BN)
        
        DoneOrNot,Value = checkStateDoneOrNot(Key, CAOrNot, AllStateRecord[i,2],BN)
        if DoneOrNot ==2:
            pass # current state is done.
            
        if DoneOrNot ==1:
            CurrentError = abs(Value - AllStateRecord[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateRecord[i,0] = Value
            AllStateRecord[i,1] = -2    # use the special number -2 to denote the confidence is reached
            AllStateRecord[i,2] = 2
            
        
        if DoneOrNot ==0:
            NewStateValue,NewDone,OptimalAction = updateOneState(Key,CAOrNot,BN,AllStateRecord)
            CurrentError = abs(NewStateValue - AllStateRecord[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateRecord[i,0] = NewStateValue
            AllStateRecord[i,1] = OptimalAction
            AllStateRecord[i,2] = NewDone
        
    return MaxError,AllStateRecord


def updateOneState(Key,CAOrNot,BN,AllStateRecord):
    AllCAState_List = Key[0:(BN.VSN-1)]
    AllVAState_List = Key[(BN.VSN-1):BN.VEN]
    
    NewDone = 0

    if CAOrNot == 0:
        ActionList = [(i+(BN.VSN-1)) for i, e in enumerate(AllVAState_List) if e == 0]
        ActionList = [-1]+ActionList            # plus the action (no VA: -1) as the first action
        ActionCount = len(ActionList)
            
        CandidateValueList = np.zeros((2,ActionCount))    
        
        for i in range(ActionCount):
            Action = ActionList[i]
            ActionCost = BN.getVACost(Action)            
                
            # get the next node keys
            NumOfNextState, NextNodeKeys, Next_CAOrNot = getNextStateKeys(BN,CAOrNot,Key,Action) # NextNodeKeys is a LIST
            
            if NumOfNextState == 2:      
                Next_Index_1 = changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0], BN)
                Next_Value_1 = AllStateRecord[Next_Index_1,0]
                Next_Done_1 = AllStateRecord[Next_Index_1,2]
                
                Next_Index_2 = changeKey2Index(NextNodeKeys[1], Next_CAOrNot[1], BN)
                Next_Value_2 = AllStateRecord[Next_Index_2,0]
                Next_Done_2 = AllStateRecord[Next_Index_2,2]
                FailureCost = BN.getFailCost(Action)   
                
                ActionProb = BN.obtainTestNodeProb(Key, Action)     # prob for the true result   
                CandidateValueList[0,i] = -ActionCost + ActionProb*Next_Value_1+(1-ActionProb)*(Next_Value_2-FailureCost)
                
                if Next_Done_1 > 0 and Next_Done_2 > 0:
                    CandidateValueList[1,i] = 2
                    
            if NumOfNextState == 1:     # current action is -1
                CandidateValueList[0,i] = 0
                CandidateValueList[1,i] = 2
            
        Temp = np.where(CandidateValueList[0,:]==np.max(CandidateValueList[0,:]))
        OptimalActionIndex = Temp[0][0]
        OptimalAction = ActionList[OptimalActionIndex]     
        
        NewValue = CandidateValueList[0,OptimalActionIndex]
        if np.min(CandidateValueList[1,:]) > 0:
            NewDone = 2


    # current activity is CA and the result of last VA is false            
    elif CAOrNot == 1:
        ActionList = [i for i, e in enumerate(AllCAState_List) if e == 0]
        ActionList = [-1]+ActionList            # plus the action (no VA: -1) as the first action
        ActionCount = len(ActionList)
        
        CandidateValueList = np.zeros((2,ActionCount)) 
            
        for i in range(ActionCount):
            Action = ActionList[i]
            ActionCost = BN.getCACost(Action)
            
            # get the next node keys
            NumOfNextState, NextNodeKeys,Next_CAOrNot = getNextStateKeys(BN,CAOrNot,Key,Action) # NextNodeKeys is a LIST
            
            if NumOfNextState == 1:                
                Next_Index = changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0], BN)
                Next_Value = AllStateRecord[Next_Index,0]
                Next_Done = AllStateRecord[Next_Index,2]
                
                CandidateValueList[0,i] = -ActionCost + Next_Value
                if Next_Done > 0:
                    CandidateValueList[1,i] = 2
                    
        Temp = np.where(CandidateValueList[0,:]==np.max(CandidateValueList[0,:]))
        OptimalActionIndex = Temp[0][0]
        OptimalAction = ActionList[OptimalActionIndex]
        
        NewValue = CandidateValueList[0,OptimalActionIndex] 
        if np.min(CandidateValueList[1,:]) > 0:
            NewDone = 2
        
    return NewValue,NewDone,OptimalAction
        
# for state update
def getNextStateKeys(BN, CAOrNot, Key, Action):     
    # node is state
    # current activity is VA
    if CAOrNot == 0:
        if Action == -1 or Action == -2:
            Key_1 = Key.copy()
            NumOfNextState = 0
            NextNodeKeys = [Key_1]
            Next_CAOrNot = [0]
        elif Action >= (BN.VSN-1):
            NumOfNextState = 2
            Key_1 = Key.copy()
            Key_1[Action] = 1
            Key_2 = Key.copy()
            Key_2[Action] = -1
            
            NextNodeKeys = [Key_1, Key_2]
            # if current is VA and its result is true,the next CA state is skipped and go to the next VA instead.
            Next_CAOrNot = [0,1]
        
        else:
            print('Error of Action!')
            print(Action)
        
    # current activity is CA and the result of last VA is true  
    #if NodeType == 1:
    #    NumOfNextState = 1            
    #    Key = NodeKey.copy()
    #    NextNodeKeys = [Key]
            
    # current activity is CA and the result of last VA is false     
    if CAOrNot == 1:
        Key_1 = Key.copy()
        if Action == -1:    # no CA
            NumOfNextState = 1    
            NextNodeKeys = [Key_1]
            Next_CAOrNot = [0]
        elif Action >= 0 and Action < (BN.VSN-1):   # CA   
            ChildList = BN.findAllChildNode(Action)
            for i in ChildList:
                if i >= (BN.VSN-1):
                    Key_1[i] = 0
                
            Key_1[Action] = 1                
            NumOfNextState = 1    
            NextNodeKeys = [Key_1]    # need update!
            Next_CAOrNot = [0]
                
        else:
            print('Error of Action!')
            print(Action)    
            
    return NumOfNextState, NextNodeKeys,Next_CAOrNot

# for tree generation
def getNextStateKeys_V2(BN, CAOrNot, Key, Action):     
    # node is state
    # current activity is VA
    if CAOrNot == 0:
        if Action == -1 or Action == -2:
            Key_1 = Key.copy()
            NumOfNextState = 0
            NextNodeKeys = [Key_1]
            Next_CAOrNot = [1]
        elif Action >= (BN.VSN-1):
            NumOfNextState = 2
            Key_1 = Key.copy()
            Key_1[Action] = 1
            Key_2 = Key.copy()
            Key_2[Action] = -1
            
            NextNodeKeys = [Key_1, Key_2]
            # if current is VA and its result is true,the next CA state is skipped and go to the next VA instead.
            Next_CAOrNot = [1,1]
        
        else:
            print('Error of Action!')
            print(Action)
        
    # current activity is CA and the result of last VA is true  
    #if NodeType == 1:
    #    NumOfNextState = 1            
    #    Key = NodeKey.copy()
    #    NextNodeKeys = [Key]
            
    # current activity is CA and the result of last VA is false     
    if CAOrNot == 1:
        Key_1 = Key.copy()
        if Action == -1:    # no CA
            NumOfNextState = 1    
            NextNodeKeys = [Key_1]
            Next_CAOrNot = [0]
        elif Action >= 0 and Action < (BN.VSN-1):   # CA   
            ChildList = BN.findAllChildNode(Action)
            for i in ChildList:
                if i >= (BN.VSN-1):
                    Key_1[i] = 0
                
            Key_1[Action] = 1                
            NumOfNextState = 1    
            NextNodeKeys = [Key_1]    # need update!
            Next_CAOrNot = [0]
                
        else:
            print('Error of Action!')
            print(Action)    
            
    return NumOfNextState, NextNodeKeys,Next_CAOrNot

def checkStateDoneOrNot(Key, CAOrNot, CurrentLabel,BN):
    DoneOrNot = 0
    Value = 0
    if CurrentLabel == 2:
        DoneOrNot = 2
        
    else:
        
        AllVAState_List = Key[(BN.VSN-1):BN.VEN]
        if CAOrNot == 1 and sum(map(abs,AllVAState_List)) == 0:
            DoneOrNot = 2

        else:
            _, CheckResult, Value = BN.checkDoneOrNot(Key)
            if CheckResult == 1:
                DoneOrNot = 1
        
    return DoneOrNot,Value
    
    
#def search_strategy():
    
    
def init_ValueFunction(BN):
    StateCount = 2*pow(2,(BN.VSN-1))*pow(3,(BN.VEN-BN.VSN+1))
    
    # Col: StateValue; Action; Label: (0: not updated; 1: updated; 2: ended) 
    ValueFunction = np.zeros((StateCount,3))
    
    return StateCount,ValueFunction


def useExistingValueFunction(AllStateRecord):
    #StateCount = 2*pow(2,(BN.VSN-1))*pow(3,(BN.VEN-BN.VSN+1))
    
    # Col: StateValue; Action; Label: (0: not updated; 1: updated; 2: ended) 
    ValueFunction = AllStateRecord
    
    ValueFunction[:,1:3] = 0
    
    return ValueFunction



def changeIndex2Key(Index, BN):
    NumOfVA = BN.VEN-BN.VSN+1
    CAOrNot = 0 # 0: VA; 1: CA and previous VA is false
    Key = [0]*BN.VEN
    Index_2 = Index
    for i in range(NumOfVA):
        Rem = Index_2 - int(Index_2/3)*3
        if Rem == 1:
            Key[-i-1] = -1
        if Rem == 2:
            Key[-i-1] = 1
            
        Index_2 = int(Index_2/3)
            
    for i in range((BN.VSN-1)):
        Rem = Index_2 - int(Index_2/2)*2
        Key[-i-1-NumOfVA] = Rem
        
        Index_2 = int(Index_2/2)
        
    CAOrNot = Index_2
    
    return Key, CAOrNot

def changeKey2Index(Key, CAOrNot, BN):
    NumOfVA = BN.VEN-BN.VSN+1
    Index = 0
    
    for i in range(NumOfVA):
        Rem = 0
        if Key[-i-1] == -1:
            Rem = 1
        if Key[-i-1] == 1:
            Rem = 2
        Index = Index + Rem*pow(3,i)
        
    for i in range((BN.VSN-1)):
        Rem = Key[-i-1-NumOfVA]
        Index = Index + Rem*pow(2,i)*pow(3,NumOfVA)
        
    if CAOrNot == 1:
        Index = Index + pow(2,(BN.VSN-1))*pow(3,NumOfVA)
        
    return Index
    
def generateOptimalTree(BN,AllStateRecord):
    NodeList = []
    NoActionNodeList = []
    ListOfOptimalTree = []
    
    # initial node
    Index = 0
    TreeIndex = [0,0]
    Value = AllStateRecord[Index,0]
    
    Init_Key,Init_CAOrNot =  changeIndex2Key(Index,BN)
    InitAction = int(AllStateRecord[Index,1])
    InitNode = [Index, TreeIndex, Value,Init_Key,InitAction]
               
    NodeList.append(InitNode)
    NoActionNodeList.append(0)
    Signal= 0
    while len(NodeList)>0:
            
        CurrentNode = NodeList.pop(0)
        NoActionOrYes= NoActionNodeList.pop(0)
        ListOfOptimalTree.append(CurrentNode)
    
        CurrentNodeIndex = CurrentNode[0]
        CurrentNodeTreeIndex = CurrentNode[1]
        Current_Key,Current_CAOrNot =  changeIndex2Key(CurrentNodeIndex,BN)
        if NoActionOrYes == 0:
            CurrentAction = CurrentNode[4]
        else:
            CurrentAction = -1

        NumOfNextState, NextNodeKeys,Next_CAOrNot = getNextStateKeys_V2(BN,Current_CAOrNot,Current_Key,CurrentAction) # NextNodeKeys is a LIST
            
        if Current_CAOrNot == 0 and NumOfNextState == 2:      
            Next_Index_1 = changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0], BN)
            Next_TreeIndex_1 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_1[0] = Next_TreeIndex_1[0]+1
            Next_TreeIndex_1[1] = 2*Next_TreeIndex_1[1]
            
            # special
            Next_Index_1_temp = changeKey2Index(NextNodeKeys[0], 0, BN)
            Next_Value_1 = AllStateRecord[Next_Index_1_temp,0]  
            
            NextAction_1 = int(-1)
            
            NextNode_1 = [Next_Index_1, Next_TreeIndex_1, Next_Value_1,NextNodeKeys[0],NextAction_1]
            
            NodeList.append(NextNode_1)
            NoActionNodeList.append(1)
            
            Next_Index_2 = changeKey2Index(NextNodeKeys[1], Next_CAOrNot[1], BN)
            Next_TreeIndex_2 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_2[0] = Next_TreeIndex_2[0]+1
            Next_TreeIndex_2[1] = 2*Next_TreeIndex_2[1]+1
            Next_Value_2 = AllStateRecord[Next_Index_2,0]
            NextAction_2 = int(AllStateRecord[Next_Index_2,1])

            NextNode_2 = [Next_Index_2, Next_TreeIndex_2, Next_Value_2,NextNodeKeys[1],NextAction_2]
            
            NodeList.append(NextNode_2)
            NoActionNodeList.append(0)
            
        if Current_CAOrNot == 1:
            Next_Index_1 = changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0], BN)
            Next_TreeIndex_1 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_1[0] = Next_TreeIndex_1[0]+1
            Next_TreeIndex_1[1] = Next_TreeIndex_1[1]
            Next_Value_1 = AllStateRecord[Next_Index_1,0]
            NextAction_1 = int(AllStateRecord[Next_Index_1,1])            
            
            NextNode_1 = [Next_Index_1, Next_TreeIndex_1, Next_Value_1,NextNodeKeys[0],NextAction_1]
            
            NodeList.append(NextNode_1)
            NoActionNodeList.append(0)
            
        Signal = Signal +1 
        if Signal % 100 == 0:
            print('Signal:',str(Signal))            
            print('len(NodeList):',str(len(NodeList)))
            
        if Signal > 100:
            break
        
    return ListOfOptimalTree


def listAllActions(ListOfOptimalTree):
    MaxRow = 0
    MaxCol = 0
    
    for i in range(len(ListOfOptimalTree)):
        CurrentNode = ListOfOptimalTree[i]
        CurrentNodeTreeIndex = CurrentNode[1]
        if CurrentNodeTreeIndex[0] > MaxCol:
            MaxCol = CurrentNodeTreeIndex[0]            
        if CurrentNodeTreeIndex[1] > MaxRow:
            MaxRow = CurrentNodeTreeIndex[1]
    
    ActionTree = np.full(((MaxRow+1),(MaxCol+1)), -10)
    #ConfidenceTree = np.zeros(((MaxRow+1),(MaxCol+1)))
    
    for i in range(len(ListOfOptimalTree)):
        CurrentNode = ListOfOptimalTree[i]
        CurrentNodeTreeIndex = CurrentNode[1]
        
        #CurrentNodeIndex = CurrentNode[0]
        ActionTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = CurrentNode[4]
    
    return ActionTree



if __name__ == "__main__":
 
    # maze game
    #env = Maze()
    #global Reward_Recordlist
    
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
    start = time.clock()
    
    # basic environment
    # SYS = System()
    
    # BN model
    BN = BNet()    
    
    #
    StateCount, AllStateRecord = init_ValueFunction(BN)    
    #AllStateRecord = useExistingValueFunction(AllStateRecord)
    
    MaxError = 1000
    while MaxError > 0.01:
        MaxError,AllStateRecord = update_ValueFunction(BN,AllStateRecord)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('MaxError:',str(MaxError))
    
    
    end = time.clock()
    
    ListOfOptimalTree = generateOptimalTree(BN,AllStateRecord)
    ActionTree = listAllActions(ListOfOptimalTree)
    
    
    
    DataFileName = 'DP_TestData_0601_2.out'
    Dict = {'AllStateRecord' : AllStateRecord, 'ListOfOptimalTree' : ListOfOptimalTree\
            , 'ActionTree': ActionTree, 'VACost': BN.VACost, 'CACost': BN.CACost\
            , 'FailCost': BN.FailCost, 'RepairLRVector': BN.RepairLRVector, 'start': start, 'end': end}

    # save
    pm.saveDictVariables(Dict,DataFileName)

    # load    
    # DataFileName = './DP_TestData_0601_1.out'
    # Dict = pm.loadDictVariables(DataFileName)
    