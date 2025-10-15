# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:52:13 2021

@author: xupeng
"""


import numpy as np

#from System import System
#from BNet_model import BNet
from BNet_model_20220925 import BNet
from AllStates4ThresRuleV3 import AllStates4ThresRule

#import copy
import time

import PengMethodList as pm


def update_AllOrders(BN, AllStateSet):
    StateCount = AllStateSet.StateCount
    
    NeedIteration = 1
    TotalNum = 0
    
    while NeedIteration == 1:
        TotalNum += 1
        NeedIteration = 0
        for i in range(StateCount):
            UpdateOrNot = AllStateSet.updateOneStateOrder(i)
            
            if UpdateOrNot == 1:
                NeedIteration = 1
                        
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('NeedIteration:',str(NeedIteration))
    
    #FileName = 'DP_OrderData_bak.out'
    #Dict = {'AllStateOrders': AllStateSet.AllStateOrders, 'TotalNum': TotalNum}

    # save
    #pm.saveDictVariables(Dict,FileName)
            
    return TotalNum
    

def update_ValueFunction(BN, AllStateSet):
    MaxError = 0
    
    Sequence = np.argsort(-AllStateSet.AllStateOrders, axis=0)    # sort in descending sequence
    
    IndexForMonitor = 0
    for ii in Sequence:
        i = ii[0]
        
        Key,CAOrNot =  AllStateSet.changeIndex2Key(i)
        
        DoneOrNot,Value = checkStateDoneOrNot(Key, CAOrNot, AllStateSet.AllStateLabels[i,0],BN)
        if DoneOrNot ==2:
            pass # current state is done.
            
        if DoneOrNot ==1:
            CurrentError = abs(Value - AllStateSet.AllStateValues[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateSet.AllStateValues[i,0] = Value
            AllStateSet.AllStateActions[i,0] = -2    # use the special number -2 to denote the confidence is reached
            AllStateSet.AllStateLabels[i,0] = 2
            
        
        if DoneOrNot == 0:
            NewStateValue,NewDone,OptimalAction = updateOneState(Key,CAOrNot,BN,AllStateSet)
            CurrentError = abs(NewStateValue - AllStateSet.AllStateValues[i,0])
            if CurrentError > MaxError:
                MaxError = CurrentError
                
            AllStateSet.AllStateValues[i,0] = NewStateValue
            AllStateSet.AllStateActions[i,0] = OptimalAction
            AllStateSet.AllStateLabels[i,0] = NewDone
        
        IndexForMonitor += 1
        if IndexForMonitor % 5000 == 0:            
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            print('IndexForMonitor:',str(IndexForMonitor))
            
        if IndexForMonitor % 100000 == 0:
            FileName = 'DP_Record_Benchmark_0730_'+str(IndexForMonitor)+'_bakupdata.out'
            AllStateSet.saveAllStates(FileName)
            print('Saved for:',str(IndexForMonitor))
            
    return MaxError,AllStateSet


def updateOneState(Key,CAOrNot,BN,AllStateSet):
    #AllCAState_List = Key[0:(BN.VSN-1)]
    #AllVAState_List = Key[(BN.VSN-1):BN.VEN]
    
    NewDone = 0

    if CAOrNot == 0:
        #ActionList = [(i+(BN.VSN-1)) for i, e in enumerate(AllVAState_List) if e == 0]
                
        TestKey = AllStateSet.getTestKey(Key)
        ActionList = [(i+(BN.VSN-1)) for i, e in enumerate(TestKey) if e == 0]        
        ActionList = [-1]+ActionList            # plus the action (no VA: -1) as the first action
        ActionCount = len(ActionList)
            
        CandidateValueList = np.zeros((2,ActionCount))    
        
        for i in range(ActionCount):
            Action = ActionList[i]
            ActionCost = BN.getVACost(Action)            
                
            # get the next node keys
            NumOfNextState, NextNodeKeys, Next_CAOrNot,ReworkOrNot = AllStateSet.getNextStateKeys(CAOrNot,Key,Action) # NextNodeKeys is a LIST
            
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
                
                if ReworkOrNot[1] == 1: # reworked
                    CACost = BN.getCACost4ThresRule(Action)
                    CandidateValueList[0,i] = CandidateValueList[0,i] + (1-ActionProb)*(-CACost)
                                    
                
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
        
    return NewValue,NewDone,OptimalAction
        


def checkStateDoneOrNot(Key, CAOrNot, CurrentLabel,BN):
    DoneOrNot = 0
    Value = 0
    if CurrentLabel == 2:
        DoneOrNot = 2
        
    else:
        TestKey = AllStateSet.getTestKey(Key)
        AllVAState_List = Key[(BN.VSN-1):BN.VEN]
        if CAOrNot == 1 and sum(map(abs,AllVAState_List)) == 0:
            DoneOrNot = 2
            
            print('Error of Key!')
            print(Key)
            
        elif max(TestKey) == 2:
            DoneOrNot = 2
            
        else:
            _, CheckResult, Value = BN.checkDoneOrNot(Key)
            if CheckResult == 1:
                DoneOrNot = 1
        
    return DoneOrNot,Value
    

def useExistingOrders(AllStateSet):
    DataFileName = './DP_Order_Record_20210730_bakupdata.out'
    Dict = pm.loadDictVariables(DataFileName)
    AllStateSet.AllStateOrders = Dict['AllStateOrders']
    TotalNum = Dict['TotalNum']
    
    return TotalNum


def generateOptimalTree(BN,AllStateSet):
    NodeList = []
    #NoActionNodeList = []
    ListOfOptimalTree = []
        
    # initial node
    Index = 0
    TreeIndex = [0,0]
    Value = AllStateSet.AllStateValues[Index,0]
    
    Init_Key,Init_CAOrNot =  AllStateSet.changeIndex2Key(Index)
    InitAction = int(AllStateSet.AllStateActions[Index,0])
    InitNode = [Index, TreeIndex, Value,Init_Key,InitAction]
               
    NodeList.append(InitNode)
    #NoActionNodeList.append(0)
    Signal= 0
    while len(NodeList)>0:
            
        CurrentNode = NodeList.pop(0)
        #NoActionOrYes= NoActionNodeList.pop(0)
        ListOfOptimalTree.append(CurrentNode)
    
        CurrentNodeIndex = CurrentNode[0]
        CurrentNodeTreeIndex = CurrentNode[1]
        Current_Key,Current_CAOrNot =  AllStateSet.changeIndex2Key(CurrentNodeIndex)
        CurrentAction = CurrentNode[4]
        
        #if NoActionOrYes == 0:
        #    CurrentAction = CurrentNode[4]
        #else:
        #    CurrentAction = -1

        #NumOfNextState, NextNodeKeys,Next_CAOrNot = AllStateSet.getNextStateKeys(Current_CAOrNot,Current_Key,CurrentAction) # NextNodeKeys is a LIST
        NumOfNextState, NextNodeKeys, Next_CAOrNot, ReworkOrNot = AllStateSet.getNextStateKeys(Current_CAOrNot,Current_Key,CurrentAction) # NextNodeKeys is a LIST
                       
        if Current_CAOrNot == 0 and NumOfNextState == 2:      
            Next_Index_1 = AllStateSet.changeKey2Index(NextNodeKeys[0], Next_CAOrNot[0])
            Next_TreeIndex_1 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_1[0] = Next_TreeIndex_1[0]+1
            Next_TreeIndex_1[1] = 2*Next_TreeIndex_1[1]
            Next_Value_1 = AllStateSet.AllStateValues[Next_Index_1,0]              
            NextAction_1 = int(AllStateSet.AllStateActions[Next_Index_1,0])
            
            NextNode_1 = [Next_Index_1, Next_TreeIndex_1, Next_Value_1,NextNodeKeys[0],NextAction_1]
            
            NodeList.append(NextNode_1)
            #NoActionNodeList.append(1)
            
            Next_Index_2 = AllStateSet.changeKey2Index(NextNodeKeys[1], Next_CAOrNot[1])
            Next_TreeIndex_2 = CurrentNodeTreeIndex.copy()
            Next_TreeIndex_2[0] = Next_TreeIndex_2[0]+1
            Next_TreeIndex_2[1] = 2*Next_TreeIndex_2[1]+1
            Next_Value_2 = AllStateSet.AllStateValues[Next_Index_2,0]
            NextAction_2 = int(AllStateSet.AllStateActions[Next_Index_2,0])

            NextNode_2 = [Next_Index_2, Next_TreeIndex_2, Next_Value_2,NextNodeKeys[1],NextAction_2]
            
            NodeList.append(NextNode_2)
            #NoActionNodeList.append(0)
            
            
        Signal = Signal +1 
        if Signal % 100 == 0:
            print('Signal:',str(Signal))            
            print('len(NodeList):',str(len(NodeList)))
            
        if Signal > 100:
            break
        
    return ListOfOptimalTree


def listAllActions(BN, ListOfOptimalTree):
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
    SecondConfidenceTree = np.zeros(((MaxRow+1),(MaxCol+1)))
    CurrentCostTree_1 = np.zeros(((MaxRow+1),(MaxCol+1)))
    CurrentCostTree_2 = np.zeros(((MaxRow+1),(MaxCol+1)))
    ReworkNoteTree = np.zeros(((MaxRow+1),(MaxCol+1)))
    
    for i in range(len(ListOfOptimalTree)):
        CurrentNode = ListOfOptimalTree[i]
        CurrentNodeTreeIndex = CurrentNode[1]
        CurrentNodeKey = CurrentNode[3]
        CurrentAction = CurrentNode[4]
        
        ActionTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = CurrentNode[4]
        TargetProb = BN.obtainTestNodeProb(CurrentNodeKey, BN.TargetNode)
        ConfidenceTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = TargetProb
                
        CurrentNodeIndex = CurrentNode[0]        
        Current_Key,Current_CAOrNot =  AllStateSet.changeIndex2Key(CurrentNodeIndex)
        FalseKey = AllStateSet.getFalseResultKey(Current_CAOrNot,Current_Key,CurrentAction)
        FalseProb = BN.obtainTestNodeProb(FalseKey, BN.TargetNode)
        SecondConfidenceTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = FalseProb
        
        
        if TargetProb >= BN.UpperThres:
            CurrentCostTree_1[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = BN.Revenue*TargetProb
            CurrentCostTree_2[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = BN.Revenue*TargetProb
            
        else:
            VACost = BN.getVACost(CurrentAction)  
            CurrentCostTree_1[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = VACost
            if CurrentAction >= (BN.VSN-1) and CurrentAction <= (BN.VEN-1):
                FailureCost = BN.getFailCost(CurrentAction)
                TempKey = CurrentNodeKey.copy()
                TempKey[CurrentAction] = -1
                TargetProb = BN.obtainTestNodeProb(TempKey, BN.TargetNode)
                if TargetProb <= BN.LowerThres:
                    ReworkNoteTree[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = 1
                    CACost = BN.getCACost4ThresRule(CurrentAction)            
                    CurrentCostTree_2[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = VACost+FailureCost+CACost
                else:
                    CurrentCostTree_2[CurrentNodeTreeIndex[1],CurrentNodeTreeIndex[0]] = VACost+FailureCost
                
    
    return ActionTree,ConfidenceTree,SecondConfidenceTree,CurrentCostTree_1,CurrentCostTree_2,ReworkNoteTree



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
    #StateCount, AllStateRecord = init_ValueFunction(BN)    
    #AllStateRecord = useExistingValueFunction(AllStateRecord)
    
    AllStateSet = AllStates4ThresRule(BN = BN)
    
    TotalNum = update_AllOrders(BN, AllStateSet)      # 3hour 45min 48sec
    #TotalNum = useExistingOrders(AllStateSet)
        
    MaxError = 1000
    while MaxError > 0.01:
        MaxError,AllStateSet = update_ValueFunction(BN,AllStateSet)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('MaxError:',str(MaxError))
        
        ListOfOptimalTree = generateOptimalTree(BN,AllStateSet)
        ActionTree,ConfidenceTree,SecondConfidenceTree,CurrentCostTree_1,CurrentCostTree_2,ReworkNoteTree = listAllActions(BN,ListOfOptimalTree)
    
        print(ActionTree)
        print(ListOfOptimalTree[0][2])
    
    end = time.clock()
    
    ListOfOptimalTree = generateOptimalTree(BN,AllStateSet)
    ActionTree,ConfidenceTree,SecondConfidenceTree,CurrentCostTree_1,CurrentCostTree_2,ReworkNoteTree = listAllActions(BN,ListOfOptimalTree)
    
    
    
    DataFileName = 'DP_benchmark_thresrule_20221001_LH09.out'
    Dict = {'AllStateValue' : AllStateSet.AllStateValues, 'ListOfOptimalTree' : ListOfOptimalTree, 'TotalNum' : TotalNum\
            , 'AllStateOrders': AllStateSet.AllStateOrders, 'AllStateActions': AllStateSet.AllStateActions, 'AllStateLabels': AllStateSet.AllStateLabels\
            , 'ActionTree': ActionTree, 'VACost': BN.VACost, 'CACost': BN.CACost, 'FactorMatrix':BN.FactorMatrix_np\
            , 'FailCost': BN.FailCost, 'RepairLRVector': BN.RepairLRVector, 'start': start, 'end': end}

    # save
    pm.saveDictVariables(Dict,DataFileName)

    # load    
    # DataFileName = './DP_benchmark_thresrule_0714_LH02.out'
    # Dict = pm.loadDictVariables(DataFileName)