# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:56:51 2021

@author: xp
"""


import numpy as np

from TreeNode import TreeNode
from NodeMemory import NodeMemory

class CurrentTree(object):
    
    def __init__(self, **kw):          
        self.VEN = kw.get('VEN')  
        #self._build_Tree()
        
        
    def getCurrentOptimalTree(self,BN,DepthLimit):
        
        NodeList = []
        CurrentOptimalTree = [] 
        
        # initial node key
        InitKey = [0]*(self.VEN)
                        
        ExistOrNot,MemoryIndex,InitNode = NodeMemory.checkNodeFromMemory(InitKey)
        NodeList.append(InitNode)
            
        ListIndexInOptimalTree = 0
        while len(NodeList)>0:
            
            CurrentNode = NodeList.pop(0)
        
            if ExistOrNot == 1:
                CurrentNode = NodeMemory.MemoryNodeList[MemoryIndex]
            else:
                CurrentNode = []
                print('Error 3 in getCurrentOptimalTree!')
            
            # for debug
            if ListIndexInOptimalTree == 1000:
                print('Yes')
            
            
            # update the pair of parent-child list indexes         
            CurrentNode.ListIndexInTree = ListIndexInOptimalTree
            
            ParentNode_ListIndexes = CurrentNode.ParentNodeListIndexInTree
            if len(ParentNode_ListIndexes) == 1:
                ParentNode = CurrentOptimalTree[ParentNode_ListIndexes[0]]
                ParentNode.ChildNodeListIndexesInTree.append(ListIndexInOptimalTree)
            elif len(ParentNode_ListIndexes) > 1:
                print('Error 1 in build_MCtree')
                
                
            # get optimal action
            CurrentAction = TreeNode.getOptimalAction(CurrentNode)            
            NumOfNextState, NextNodeKeys = self.getNextStateKeys(BN, CurrentNode.NodeKey, CurrentAction)
                        
            
            # expand optimal tree
            CurrentOptimalTree.append(CurrentNode)
            ListIndexInOptimalTree = ListIndexInOptimalTree + 1
            
            if NumOfNextState == 2:
                # NextNodes is a LIST of next nodes
                
                ExistOrNot_1,MemoryIndex_1,NextNode_1 = NodeMemory.checkNodeFromMemory(NextNodeKeys[0])
                NextNode_1.ParentNodeListIndexInTree = [CurrentNode.ListIndexInTree]
                if ExistOrNot_1 == 1:
                    NodeList.append(NextNode_1)
                else:                        
                    print('Error 31 in getCurrentOptimalTree!')
                ExistOrNot_2,MemoryIndex_2,NextNode_2 = NodeMemory.checkNodeFromMemory(NextNodeKeys[1])
                NextNode_2.ParentNodeListIndexInTree = [CurrentNode.ListIndexInTree]
                if ExistOrNot_2 == 0:
                    NodeList.append(NextNode_2)
                else:  
                    print('Error 32 in getCurrentOptimalTree!')
                    
        return CurrentOptimalTree
        
    
    def getNextStateKeys(self, BN, Key, Action):
        if len(Action) < 2:
            print('Error 1 in getNextStateKeys')
            
        # Stage 1: CA
        Key_Stage1 = Key.copy()
        CA_Action = Action[0]
        if CA_Action = -1:
            pass
        elif CA_Action >= 0 and CA_Action < (BN.VSN-1):   # CA   
            ChildList = BN.findAllChildNode(CA_Action)
            for i in ChildList:
                if i >= (BN.VSN-1):
                    Key_Stage1[i] = 0
                
            Key_Stage1[CA_Action] = 1        
                
        else:
            print('Error 3 in getNextStateKeys!')
            print(Action)
            
            
        # Stage 2: VA
        NodeConfidence, DoneOrNot, StateValue = BN.checkDoneOrNot(Key_Stage1)
        VA_Action = Action[1]
        if DoneOrNot == 1:
            NumOfNextState = 1    
            NextNodeKeys = [Key_Stage1]
            
        elif VA_Action == -1:
            NumOfNextState = 0    
            NextNodeKeys = []
            
        else：
            if Key1_Stage2[VA_Action] != 0:
                print('Error 4 in getNextStateKeys!')
            
            NumOfNextState = 2
            Key1_Stage2 = Key_Stage1.copy()
            Key1_Stage2[VA_Action] = -2
            Key2_Stage2 = Key_Stage1.copy()
            Key2_Stage2[VA_Action] = -1
            NextNodeKeys = [Key1_Stage2,Key2_Stage2]
            
        return NumOfNextState, NextNodeKeys
    
    
    