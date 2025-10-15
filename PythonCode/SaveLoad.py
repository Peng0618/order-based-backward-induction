# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:25:05 2020

@author: xp
"""


import shelve

filename='E:\GoogleDrive\PythonScripts\BayesNet\DQL\ValueConstants.out'

filename='E:\GoogleDrive\PythonScripts\BayesNet\DQL\RLData0213.out'

filename='D:\GoogleDrive\PythonScripts\BayesNet\DQL_0326\TestData0326.out'

# save
my_shelf = shelve.open(filename,'n') # 'n' for new
# with shelve.open(filename,'n') as my_shelf:
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    #except TypeError:
    #    #
    #    # __builtins__, my_shelf, and imported modules can not be shelved.
    #    #
    #    print('ERROR shelving: {0}'.format(key))
    except:
        print('Save Error ERROR:'.format(key))
        
my_shelf.close()


# load
my_shelf = shelve.open(filename)
for key in my_shelf:
    try:
        globals()[key]=my_shelf[key]
    except:
        print('Save Error ERROR:'.format(key))
my_shelf.close()





#%%

# input constants

ActivityCost = [350,800,300,250,300,350,350,550,450]

ReworkCost = [36620,3230,39420,6190,39010,4970,2570,39550,5650]

Revenue = 20000

UpperThres = 0.95

LowerThres = [0.2, 0.2, 0.2, 0.2, 0.2]



import pickle
 
t = []
for i in range(2):
    inp = input()
    t.append(inp)
 
with open('./the_first_pickle.pickle','w') as p:
    pickle.dump(a2,p)   #将列表t保存起来
    

pickle.dump(RL, open("tmp.txt", "w"))
 
with open('./the_first_pickle.pickle','r') as r:

    
    
    
    
    
    
    
    
    
    
    
#%%    
    

# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    
import tensorflow as tf    
saver = tf.train.Saver()
saver.save(RL.sess, './my_test_model')

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my_test_model.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))


from BNet_env import BNet
from RL_brain_DQL import DeepQNetwork
env = BNet()

RL = DeepQNetwork(env.n_actions, env.n_features,
                      env.VSN, env.N, env.TargetNode, env.TimeLimit,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter = 500,
                      memory_size=1000,
                      e_greedy_increment = 0.01
                      # output_graph=True
                      )

RL.sess = tf.Session()
RL.sess = sess
    
    
    
import time
saved_model_path = "./saved_models/{}".format(int(time.time()))

tf.keras.experimental.export_saved_model(model, saved_model_path)
saved_model_path

!ls saved_models/


new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()


with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.save(RL.sess, "./model.ckpt") 

import tensorflow as tf
import numpy as np
reader = tf.train.NewCheckpointReader('./model.ckpt')
all_variables = reader.get_variable_to_shape_map()

w1 = reader.get_tensor("eval_net/l1/w1")
w2 = reader.get_tensor("eval_net/l2/w2")
b1 = reader.get_tensor("eval_net/l1/b1")
b2 = reader.get_tensor("eval_net/l2/b2")

print(type(w1))
print(w1.shape)
print(w1[0])
w1_t = reader.get_tensor("target_net/l1/w1")
print(type(w1_t))
print(w1_t.shape)
print(w1_t[0])
{'target_net/l2/w2': [15, 10],
 'target_net/l2/b2': [1, 10],
 'target_net/l1/b1': [1, 15],
 'target_net/l1/w1': [9, 15],
 
 'eval_net/l1/b1': [1, 15],
 'eval_net/l1/w1': [9, 15],
 'eval_net/l2/b2': [1, 10],
 'eval_net/l2/w2': [15, 10],
 
 'train/eval_net/l1/w1/RMSProp': [9, 15],
 'train/eval_net/l1/b1/RMSProp': [1, 15],
 'train/eval_net/l2/b2/RMSProp': [1, 10],
 'train/eval_net/l2/w2/RMSProp': [15, 10],
 
 'train/eval_net/l1/b1/RMSProp_1': [1, 15],
 'train/eval_net/l1/w1/RMSProp_1': [9, 15],
 'train/eval_net/l2/b2/RMSProp_1': [1, 10],
 'train/eval_net/l2/w2/RMSProp_1': [15, 10]}





#%%  Load data from .spydata file
from spyderlib.utils.iofuncs import load_dictionary

globals().update(load_dictionary(fpath)[0])
data = load_dictionary(fpath)



# Save data to .spydata file
from spyder.utils.iofuncs import save_dictionary
def globalsfiltered(d):
    from spyder.widgets.dicteditorutils import globalsfilter
    from spyder.plugins.variableexplorer import VariableExplorer
    from spyder.baseconfig import get_conf_path, get_supported_types

    data = globals()
    settings = VariableExplorer.get_settings()

    get_supported_types()
    data = globalsfilter(data,                   
                         check_all=True,
                         filters=tuple(get_supported_types()['picklable']),
                         exclude_private=settings['exclude_private'],
                         exclude_uppercase=settings['exclude_uppercase'],
                         exclude_capitalized=settings['exclude_capitalized'],
                         exclude_unsupported=settings['exclude_unsupported'],
                         excluded_names=settings['excluded_names']+['settings','In'])
    return data

def saveglobals(filename):
    data = globalsfiltered()
    save_dictionary(data,filename)


#

savepath = 'test.spydata'

saveglobals(savepath) 



#%%

import shelve

T='Hiya'
val=[1,2,3]

filename='/shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()






#%%

# save class

import pickle



globals()


rw = class()

output_hal = open("1.pkl", 'wb')
str = pickle.dumps(rw)
output_hal.write(str)
output_hal.close()
        
        
        
rq = class()
with open("1.pkl",'rb') as file:
    rq  = pickle.loads(file.read())
    
    
    
    
    

#%%

import shelve

T='Hiya'
val=[1,2,3]

filename='./shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
    


my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
    
    
#%%


import pickle
Dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}

filename='D:\GoogleDrive\PythonScripts\BayesNet\DQL_0326\TestData0326.p'

with open(filename,"wb") as f:
    pickle.dump(Dict,f)

with open(filename,'rb') as f:
    Dict  = pickle.loads(f.read())
    
    
    
#%%

import shelve

T = 'Hiya'
val = [1, 2, 3]

def save_variables(globals_=None):
    if globals_ is None:
        globals_ = globals()
    filename='D:\GoogleDrive\PythonScripts\BayesNet\DQL_0326\TestData0326.out'

    my_shelf = shelve.open(filename, 'n')
    for key, value in globals_.items():
        if not key.startswith('__'):
            try:
                my_shelf[key] = value
            except Exception:
                print('ERROR shelving: "%s"' % key)
            else:
                print('shelved: "%s"' % key)
    my_shelf.close()

save_variables()

    