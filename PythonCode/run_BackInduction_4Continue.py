# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:00:38 2021

@author: xupeng
"""


import numpy as np

#from System import System
from BNet_model_20220925 import BNet
from AllStates import AllStates

#import copy
import time

import PengMethodList as pm


def update_AllOrders(BN, AllStateSet):
    StateCount = AllStateSet.StateCount
    
    NeedIteration = 1
    while NeedIteration == 1:
        NeedIteration = 0
        for i in range(StateCount):
            UpdateOrNot = AllStateSet.updateOneStateOrder(i)
            
            if UpdateOrNot == 1:
                NeedIteration = 1
        
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('NeedIteration:',str(NeedIteration))
        
    #return NeedIteration
    

def update_ValueFunction(BN, AllStateSet):
    MaxError = 0
    
    Sequence = np.argsort(-AllStateSet.AllStateOrders, axis=0)    # sort in descending sequence
    
    Signal = 0
    for ii in Sequence:
        i = ii[0]
        
        Key,CAOrNot =  AllStateSet.changeIndex2Key(i)
        
        DoneOrNot,Value = checkStateDoneOrNot(Key, CAOrNot, AllStateSet.AllStateLabels[i,0],BN)
        if DoneOrNot ==2:
            #if Signal % 5000 == 0:
            #   print('Signal:',str(Signal))  
            pass # current state is done.
            
        if DoneOrNot ==1:
            CurrentError = abs(Value - AllStateSet.AllStateValues[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateSet.AllStateValues[i,0] = Value
            AllStateSet.AllStateActions[i,0] = -2    # use the special number -2 to denote the confidence is reached
            AllStateSet.AllStateLabels[i,0] = 2
            
        
        if DoneOrNot ==0:
            NewStateValue,NewDone,OptimalAction = updateOneState(Key,CAOrNot,BN,AllStateSet)
            CurrentError = abs(NewStateValue - AllStateSet.AllStateValues[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateSet.AllStateValues[i,0] = NewStateValue
            AllStateSet.AllStateActions[i,0] = OptimalAction
            AllStateSet.AllStateLabels[i,0] = NewDone
        
        Signal = Signal +1 
        if Signal % 5000 == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print('Signal:',str(Signal))            
            
        if Signal % 50000 == 0:
            FileName = 'DP_Record_MidNet_20220925_'+str(Signal)+'_bakupdata_1.out'
            AllStateSet.saveAllStates(FileName)
            print('Saved for:',str(Signal))
            
    return MaxError,AllStateSet,Signal


def getCurrentIndexNum(BN, AllStateSet):
    Sequence = np.argsort(-AllStateSet.AllStateOrders, axis=0)    # sort in descending sequence
    
    CurrentIndexNum = 0
    for ii in Sequence:
        i = ii[0]
        
        if AllStateSet.AllStateLabels[i,0] == 2:
            CurrentIndexNum = CurrentIndexNum + 1   
        else:
            break
           
    return CurrentIndexNum

def getCurrentIndexNumFromAction(BN, AllStateSet):
    Sequence = np.argsort(-AllStateSet.AllStateOrders, axis=0)    # sort in descending sequence
    
    CurrentIndexNum = 0
    for ii in Sequence:
        i = ii[0]
                
        if AllStateSet.AllStateActions[i,0] > -10:
            CurrentIndexNum = CurrentIndexNum + 1   
        else:
            break
           
    return CurrentIndexNum


def getCurrentIndexNumFromValue(BN, AllStateSet):
    Sequence = np.argsort(-AllStateSet.AllStateOrders, axis=0)    # sort in descending sequence
    
    CurrentIndexNum = 0
    CurrentOptIndex = 0
    TempRecord_Thres = 50000  # hyperparameter
    for ii in Sequence:
        i = ii[0]
        CurrentIndexNum = CurrentIndexNum + 1   
        
        # debug
        #if CurrentIndexNum >= 719610 and CurrentIndexNum <= 719620:
        #    print(CurrentIndexNum)
        #    print(i)
        #    print(AllStateSet.AllStateValues[i,0])
        
        if AllStateSet.AllStateValues[i,0] > 0:            
            CurrentOptIndex = CurrentIndexNum
            TempRecord = 0
        else:
            TempRecord = TempRecord + 1
            
        if TempRecord > TempRecord_Thres:
            break
           
    return CurrentOptIndex


def update_ValueFunctionWithStartIndex(BN, AllStateSet, StartIndex):
    MaxError = 0
    
    Sequence = np.argsort(-AllStateSet.AllStateOrders, axis=0)    # sort in descending sequence
    
    Signal = 0
    for ii in Sequence:
        i = ii[0]
        
        if Signal < StartIndex:
            Signal = Signal + 1
            continue
        
        Key,CAOrNot =  AllStateSet.changeIndex2Key(i)
        
        DoneOrNot,Value = checkStateDoneOrNot(Key, CAOrNot, AllStateSet.AllStateLabels[i,0],BN)
        if DoneOrNot ==2:
            pass
            
        if DoneOrNot ==1:
            CurrentError = abs(Value - AllStateSet.AllStateValues[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateSet.AllStateValues[i,0] = Value
            AllStateSet.AllStateActions[i,0] = -2    # use the special number -2 to denote the confidence is reached
            AllStateSet.AllStateLabels[i,0] = 2
            
        
        if DoneOrNot ==0:
            NewStateValue,NewDone,OptimalAction = updateOneState(Key,CAOrNot,BN,AllStateSet)
            CurrentError = abs(NewStateValue - AllStateSet.AllStateValues[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateSet.AllStateValues[i,0] = NewStateValue
            AllStateSet.AllStateActions[i,0] = OptimalAction
            AllStateSet.AllStateLabels[i,0] = NewDone
        
        Signal = Signal +1 
        if Signal % 5000 == 0:
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print('Signal:',str(Signal))  
            print('DoneOrNot:',str(DoneOrNot)) 
            
        if Signal % 50000 == 0:
            FileName = 'DP_Record_MidNet_0715_'+str(Signal)+'_bakupdata_1.out'
            AllStateSet.saveAllStates(FileName)
            print('Saved for:',str(Signal))
            
    return MaxError,AllStateSet,Signal



def updateOneState(Key,CAOrNot,BN,AllStateSet):
    AllCAState_List = Key[0:(BN.VSN-1)]
    AllVAState_List = Key[(BN.VSN-1):BN.VEN]
    
    NewDone = 1

    if CAOrNot == 0:
        ActionList = [(i+(BN.VSN-1)) for i, e in enumerate(AllVAState_List) if e == 0]
        ActionList = [-1]+ActionList            # plus the action (no VA: -1) as the first action
        ActionCount = len(ActionList)
            
        CandidateValueList = np.zeros((2,ActionCount))    
        
        for i in range(ActionCount):
            Action = ActionList[i]
            ActionCost = BN.getVACost(Action)            
                
            # get the next node keys
            NumOfNextState, NextNodeKeys, Next_CAOrNot = getNextStateKeysForValue(BN,CAOrNot,Key,Action) # NextNodeKeys is a LIST
            
            if NumOfNextState == 2:      
                Next_Index_1 = AllStateSet.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
                Next_Value_1 = AllStateSet.AllStateValues[Next_Index_1,0]
                Next_Done_1 = AllStateSet.AllStateLabels[Next_Index_1,0]
                
                Next_Index_2 = AllStateSet.changeKey2Index(NextNodeKeys[1], Next_CAOrNot[1])
                Next_Value_2 = AllStateSet.AllStateValues[Next_Index_2,0]
                Next_Done_2 = AllStateSet.AllStateLabels[Next_Index_2,0]
                FailureCost = BN.getFailCost(Action)   
                
                ActionProb = BN.obtainTestNodeProb(Key, Action)     # prob for the true result   
                CandidateValueList[0,i] = -ActionCost + ActionProb*Next_Value_1+(1-ActionProb)*(Next_Value_2-FailureCost)
                
                if Next_Done_1 > 0 and Next_Done_2 > 0:
                    CandidateValueList[1,i] = 2
                    
            if NumOfNextState == 0:     # current action is -1
                CandidateValueList[0,i] = 0
                CandidateValueList[1,i] = 2
            
        Temp = np.where(CandidateValueList[0,:]==np.max(CandidateValueList[0,:]))
        OptimalActionIndex = Temp[0][0]
        OptimalAction = ActionList[OptimalActionIndex]     
        
        NewValue = CandidateValueList[0,OptimalActionIndex]
        if np.min(CandidateValueList[1,:]) > 0:
            NewDone = 2

        # only works for backward induction:
        if OptimalAction == -1:
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
            NumOfNextState, NextNodeKeys,Next_CAOrNot = getNextStateKeysForValue(BN,CAOrNot,Key,Action) # NextNodeKeys is a LIST
            
            if NumOfNextState == 1:                
                Next_Index = AllStateSet.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
                Next_Value = AllStateSet.AllStateValues[Next_Index,0]
                Next_Done = AllStateSet.AllStateLabels[Next_Index,0]
                
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
def getNextStateKeysForValue(BN, CAOrNot, Key, Action):     
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
    
    

def generateOptimalTree(BN,AllStateSet):
    NodeList = []
    NoActionNodeList = []
    ListOfOptimalTree = []
    
    # initial node
    Index = 0
    TreeIndex = [0,0]
    Value = AllStateSet.AllStateValues[Index,0]
    
    Init_Key,Init_CAOrNot =  AllStateSet.changeIndex2Key(Index)
    InitAction = int(AllStateSet.AllStateActions[Index,0])
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
        Current_Key,Current_CAOrNot =  AllStateSet.changeIndex2Key(CurrentNodeIndex)
        if NoActionOrYes == 0:
            CurrentAction = CurrentNode[4]
        else:
            CurrentAction = -1

        NumOfNextState, NextNodeKeys,Next_CAOrNot = AllStateSet.getNextStateKeys(Current_CAOrNot,Current_Key,CurrentAction) # NextNodeKeys is a LIST
            
        if Current_CAOrNot == 0 and NumOfNextState == 2:      
            Next_Index_1 = AllStateSet.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
            Next_TreeIndex_1 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_1[0] = Next_TreeIndex_1[0]+1
            Next_TreeIndex_1[1] = 2*Next_TreeIndex_1[1]
            
            # special
            Next_Index_1_temp = AllStateSet.changeKey2Index(NextNodeKeys[0], 0)
            Next_Value_1 = AllStateSet.AllStateValues[Next_Index_1_temp,0]  
            
            NextAction_1 = int(-1)
            
            NextNode_1 = [Next_Index_1, Next_TreeIndex_1, Next_Value_1,NextNodeKeys[0],NextAction_1]
            
            NodeList.append(NextNode_1)
            NoActionNodeList.append(1)
            
            Next_Index_2 = AllStateSet.changeKey2Index(NextNodeKeys[1], Next_CAOrNot[1])
            Next_TreeIndex_2 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_2[0] = Next_TreeIndex_2[0]+1
            Next_TreeIndex_2[1] = 2*Next_TreeIndex_2[1]+1
            Next_Value_2 = AllStateSet.AllStateValues[Next_Index_2,0]
            NextAction_2 = int(AllStateSet.AllStateActions[Next_Index_2,0])

            NextNode_2 = [Next_Index_2, Next_TreeIndex_2, Next_Value_2,NextNodeKeys[1],NextAction_2]
            
            NodeList.append(NextNode_2)
            NoActionNodeList.append(0)
            
        if Current_CAOrNot == 1:
            Next_Index_1 = AllStateSet.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
            Next_TreeIndex_1 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_1[0] = Next_TreeIndex_1[0]+1
            Next_TreeIndex_1[1] = Next_TreeIndex_1[1]
            Next_Value_1 = AllStateSet.AllStateValues[Next_Index_1,0]
            NextAction_1 = int(AllStateSet.AllStateActions[Next_Index_1,0])            
            
            NextNode_1 = [Next_Index_1, Next_TreeIndex_1, Next_Value_1,NextNodeKeys[0],NextAction_1]
            
            NodeList.append(NextNode_1)
            NoActionNodeList.append(0)
            
        Signal = Signal +1 
        if Signal % 100 == 0:
            print('Signal:',str(Signal))            
            print('len(NodeList):',str(len(NodeList)))
            
        #if Signal > 100:
        #    break
        
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



def listAllActionsV2(BN, ListOfOptimalTree):
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
    ConfidenceTree = np.zeros(((MaxRow+1),(MaxCol+1)))
    CurrentCostTree_1 = np.zeros(((MaxRow+1),(MaxCol+1)))
    CurrentCostTree_2 = np.zeros(((MaxRow+1),(MaxCol+1)))
    ReworkNoteTree = np.zeros(((MaxRow+1),(MaxCol+1)))
    
    NumberOfPath = 0
    
    for i in range(len(ListOfOptimalTree)):
        CurrentNode = ListOfOptimalTree[i]
        CurrentNodeIndex = CurrentNode[0]
        __, Current_CAOrNot =  AllStateSet.changeIndex2Key(CurrentNodeIndex)
        CurrentNodeTreeIndex = CurrentNode[1]
        CurrentNodeKey = CurrentNode[3]
        CurrentAction = CurrentNode[4]
        
        if CurrentAction == -2:
            NumberOfPath = NumberOfPath + 1
            
        if Current_CAOrNot == 0 and CurrentAction == -1:
            NumberOfPath = NumberOfPath + 1
        
        #CurrentNodeIndex = CurrentNode[0]
        ActionTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = CurrentNode[4]
        TargetProb = BN.obtainTestNodeProb(CurrentNodeKey, BN.TargetNode)
        ConfidenceTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = TargetProb
        if TargetProb >= BN.UpperThres:
            CurrentCostTree_1[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = BN.Revenue*TargetProb
            CurrentCostTree_2[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = BN.Revenue*TargetProb
            
        else:
            if Current_CAOrNot == 0:
                ActionCost = BN.getVACost(CurrentAction)  
            elif Current_CAOrNot == 1:
                ActionCost = BN.getCACost(CurrentAction)  
                
            CurrentCostTree_1[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = ActionCost
            if CurrentAction >= (BN.VSN-1) and CurrentAction <= (BN.VEN-1):
                FailureCost = BN.getFailCost(CurrentAction)
                CurrentCostTree_2[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = ActionCost+FailureCost
    
    return ActionTree,ConfidenceTree,CurrentCostTree_1,CurrentCostTree_2,ReworkNoteTree,NumberOfPath


def initAllStateWithGivenValue(AllStateSet,Dict):
    AllStateSet.AllStateValues = Dict['AllStateValue']
    AllStateSet.AllStateLabels = Dict['AllStateLabels']
    AllStateSet.AllStateOrders = Dict['AllStateOrders']
    AllStateSet.AllStateActions = Dict['AllStateActions']
    

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
        
    AllStateSet = AllStates(BN = BN)
    
    update_AllOrders(BN, AllStateSet)
    
    # ExistFileName = './DP_TestData_MidNet_0614_important_1.out'
    #ExistFileName = './DP_Record_MidNet_0715_50000_bakupdata_1.out'
    #Dict = pm.loadDictVariables(ExistFileName)
    #initAllStateWithGivenValue(AllStateSet,Dict)
    
    # StartIndex = getCurrentIndexNum(BN, AllStateSet)
    # StartIndex2 = getCurrentIndexNumFromAction(BN, AllStateSet)
    #StartIndex4 = getCurrentIndexNumFromValue(BN, AllStateSet)
    
    #StartIndex4 = int(StartIndex4/100-1)*100
    # StartIndex4 = 863500  --- 863592  --- 
    
    MaxError = 1000
    while MaxError > 0.01:
        #MaxError,AllStateSet,Signal = update_ValueFunctionWithStartIndex(BN,AllStateSet,StartIndex4)
        MaxError,AllStateSet,Signal = update_ValueFunction(BN,AllStateSet)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('MaxError:',str(MaxError))
    
    
    end = time.clock()
    
    ListOfOptimalTree = generateOptimalTree(BN,AllStateSet)
    #ActionTree = listAllActions(ListOfOptimalTree)
    ActionTree,ConfidenceTree,CurrentCostTree_1,CurrentCostTree_2,ReworkNoteTree,NumberOfPath = listAllActionsV2(BN,ListOfOptimalTree)
    
    
    # DataFileName = 'DP_TestData_MidNet_0608_ordered_1.out'
    DataFileName = 'DP_TestData_MidNet_20220925.out'
    Dict = {'AllStateValue' : AllStateSet.AllStateValues, 'VACost': BN.VACost, 'CACost': BN.CACost\
            , 'AllStateOrders': AllStateSet.AllStateOrders, 'AllStateActions': AllStateSet.AllStateActions, 'AllStateLabels': AllStateSet.AllStateLabels\
            , 'ActionTree': ActionTree, 'ListOfOptimalTree' : ListOfOptimalTree\
            , 'FailCost': BN.FailCost, 'RepairLRVector': BN.RepairLRVector\
            , 'start': start, 'end': end}

    # save
    pm.saveDictVariables(Dict,DataFileName)

    # load    
    # DataFileName = './DP_TestData_MidNet_0608_ordered_1.out'
    # Dict = pm.loadDictVariables(DataFileName)