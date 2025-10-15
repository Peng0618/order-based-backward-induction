# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:02:31 2021

@author: xupeng
"""

import numpy as np



class TreeNode(object):
    __slots__ = ["VEN", "NodeKey", "NodeType", "NodeIndexInTree", \
    "ConfidenceList", "Action", "ActionCost", "ActionProb",\
    "ListIndexInTree", "ParentNodeListIndexInTree", "ChildNodeListIndexesInTree", \
    "ValueMatrix", "LockMatrix"]
    
    def __init__(self, **kw):    
        #super(TreeNode, self).__init__()
        
        #self.VSN = kw.get('VSN')
        self.VEN = kw.get('VEN')     
        
        # basic properties
        # NodeKey: -1: false result; 0: no result; 1: true result
        # in BN: 0: no result; 1: false result; 2: true result
        # in NN: 0: no result; 1: has results (true/false)
        if 'NodeKey' in kw:
            self.NodeKey = kw.get('NodeKey')
        else:
            self.NodeKey = [0]*(self.VEN)      
        
        # NodeType:       0: current activity is VA; 
        #                 1: current activity is CA and last VA is true
        #                 2: current activity is CA and last VA is false        
        self.NodeType = kw.get('NodeType',0)   
        self.NodeIndexInTree = kw.get('NodeIndexInTree',[0,0])
        
        
        # properties from BN
        self.ConfidenceList = []
        
        # value and lock vector
        self.ValueMatrix = []
        self.LockMatrix = []
        
        # action info
        self.Action = []    # 1 number [0] for a bud node, 2 numbers [CA num, VA num] for an expanded node
        self.ActionCost = []
        self.ActionProb = []
        
        # properties for tree update
        self.ListIndexInTree = []
        self.ParentNodeListIndexInTree = []
        self.ChildNodeListIndexesInTree = []
        
        #self.NodeValue = kw.get('NodeValue',[])   
        #self.NodeVisitCount = kw.get('NodeVisitCount',0)
        # ValueLabel:
        # 0: no value added yet
        # 1: use prediction value from NN/ it must be a bud node.
        # 2: use updated value from tree update
        # 5: this node meets stopping rule and its value is fixed
        #self.NodeValue_label = 0
        
        # properties for storage        
        #self.NodeIndexInMemory = kw.get('NodeIndexInMemory',0)
        #Temp = np.append(self.NodeKey,self.NodeValue)
        #self.MemoryRecord = np.append(Temp,self.NodeVisitCount)
        
        # source: https://www.liaoxuefeng.com/discuss/969955749132672/1106749266326816
        
        
        
    def getOptimalAction(Node):
        ValueMatrix = Node.ValueMatrix
        Action = Node.Action
        LockMatrix = Node.LockMatrix
        
        if len(Action) == 0:
            print('Error 1 in getOptimalAction')
            NodeValue = []
            NodeLock = []
            
        elif len(Action) == 1:  # not expanded bud
            NodeValue = ValueMatrix
            NodeLock = LockMatrix
            
        else:
            NodeValue = ValueMatrix[Action[0]][Action[1]]
            NodeLock = LockMatrix[Action[0]][Action[1]]
            
        return NodeValue,NodeLock
        
        
        