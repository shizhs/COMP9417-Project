#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# ### Split Datasets ###

# In[7]:


### Split Activities ###
def splitActivities(inputs):
    activity_list = ['Stationary', 'Walk', 'Run', 'Unknown']
    df_walk = pd.DataFrame(columns=None)
    df_run = pd.DataFrame(columns=None)
    for i in inputs['activity'].index:
        for j in inputs['activity'].columns:
            s,w,r,u = inputs['activity'].loc[i, j].split('|')
            total = int(s) + int(w) + int(r) + int(u)
            if total == 0:
                df_walk.loc[i, j] = 0
                df_run.loc[i, j] = 0
            else:
                df_walk.loc[i, j] = int(w)/total
                df_run.loc[i, j] = int(r)/total
    df_walk.to_csv('./Final_Data/walk.csv')
    df_run.to_csv('./Final_Data/run.csv')


# In[8]:


### Split Audio ###
def splitAudio(inputs):
    audio_list = ['Silence', 'Voice', 'Noise']
    df_noise = pd.DataFrame(columns=None)
    df_total_audio = pd.DataFrame(columns=None)
    for i in inputs['audio'].index:
        for j in inputs['audio'].columns:
            s,v,n = inputs['audio'].loc[i, j].split('|')
            total = int(s) + int(v) + int(n)
            if total == 0:
                df_noise.loc[i, j] = 0
            else:
                df_noise.loc[i, j] = int(n)/total
            df_total_audio.loc[i, j] = total
    df_noise.to_csv('./Final_Data/noise.csv')


# In[9]:


### Split Dark ###
def splitDark(inputs):
    df_dark_freq = pd.DataFrame(columns=None)
    df_dark_time = pd.DataFrame(columns=None)
    for i in inputs['dark'].index:
        for j in inputs['dark'].columns:
            t, f = inputs['dark'].loc[i, j].split('|')
            df_dark_time.loc[i, j] = t
            df_dark_freq.loc[i, j] = f
    df_dark_time.to_csv('./Final_Data/dark_time.csv')
    df_dark_freq.to_csv('./Final_Data/dark_freq.csv')


# In[10]:


### Split Conversation ###
def splitConversation(inputs):
    df_conversation_freq = pd.DataFrame(columns=None)
    df_conversation_time = pd.DataFrame(columns=None)
    for i in inputs['conversation'].index:
        for j in inputs['conversation'].columns:
            t, f = inputs['conversation'].loc[i, j].split('|')
            df_conversation_time.loc[i, j] = t
            df_conversation_freq.loc[i, j] = f
    df_conversation_time.to_csv('./Final_Data/conversation_time.csv')
    df_conversation_freq.to_csv('./Final_Data/conversation_freq.csv')


# In[ ]:




