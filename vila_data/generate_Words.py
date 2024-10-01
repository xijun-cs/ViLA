import torch 
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import os
import re

##
from s3dg import S3D
net = S3D('s3d_dict.npy', 512)  
net.load_state_dict(th.load('s3d_howto100m.pth'))
words = np.load('words60k.npy')
## the above needs to be obtained from https://github.com/antoine77340/MIL-NCE_HowTo100M

net = net.eval()

text_output = net.text_module(words)
text_embedding = text_output['text_embedding']

vid_dict = torch.load('s3d_ActivityNet.pth')
pool=torch.nn.MaxPool1d(5)


def s3d_words (vid_id, k):
    word_list=[]
    m5c = vid_dict[str(vid_id)]
    if len(m5c) < 30:
        m5c = F.interpolate(m5c.unsqueeze(0).transpose(1,2).contiguous(), 30).transpose(1,2).contiguous().squeeze(0)
    else:
        m5c = F.interpolate(m5c.unsqueeze(0).transpose(1,2).contiguous(), 30, mode='linear',align_corners=False).transpose(1,2).contiguous().squeeze(0)
    video_embedding = net.fc(m5c)
    all_scores = th.matmul(text_embedding2, video_embedding.t())
    all_scores = pool(all_scores.unsqueeze(0)).squeeze(0)
    w = all_scores.topk(k,dim=0)[1].T.flatten().tolist()
    for i in w:
        word_list.append(words[i])
        
    return word_list
k=15
df = pd.read_csv('../../data/iVQA/train.csv')
for i in range(0,len(df)): #for i, row in df.iterrows():
    v_id = df.loc[[i]]['video_id'].to_string(index=False)
    if v_id[0] == " ": 
        v_id = v_id[1:]
    question = df.loc[[i]]['question'].to_string(index=False)
    q = question + '? '
    word_list = s3d_words(v_id,k)
    for j in range(0, len(word_list)):
        q +=  word_list[j] + ", "
    df.at[i,'question']=q
    print(i)
df.to_csv("train.csv",index=False)  
df = pd.read_csv('../../data/iVQA/val.csv')
for i in range(0,len(df)): #for i, row in df.iterrows():
    v_id = df.loc[[i]]['video_id'].to_string(index=False)
    if v_id[0] == " ": 
        v_id = v_id[1:]
    question = df.loc[[i]]['question'].to_string(index=False)
    q = question + '? '
    word_list = s3d_words(v_id,k)
    for j in range(0, len(word_list)):
        q +=  word_list[j] + ", "
    df.at[i,'question']=q
    print(i)
df.to_csv("val.csv",index=False)  
df = pd.read_csv('../../data/iVQA/test.csv')
for i in range(0,len(df)): #for i, row in df.iterrows():
    v_id = df.loc[[i]]['video_id'].to_string(index=False)
    if v_id[0] == " ": 
        v_id = v_id[1:]
    question = df.loc[[i]]['question'].to_string(index=False)
    q = question + '? '
    word_list = s3d_words(v_id,k)
    for j in range(0, len(word_list)):
        q +=  word_list[j] + ", "
    df.at[i,'question']=q
    print(i)
df.to_csv("test.csv",index=False)