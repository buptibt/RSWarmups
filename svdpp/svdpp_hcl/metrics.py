# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:26:38 2019

@author: HP-PC
"""
import math
def dcg(mode,rels,k):
    # dcg@k   k>=1
    # rels.shape= (len_rels, )   relevance scores of recommendated documents
    if mode==1:
        gain = rels[0]
        for i in range(1,k):
            gain += rels[i]/math.log2(i+1)
            
    elif mode==2:
        gain = 0
        for i in range(k):
            gain += ( pow(2,rels[i])-1 )/math.log2(i+2)
    
    else:
        r = rels[:k]
        r = [x/math.log2(i+2) for i,x in enumerate(r)]
        gain = sum(r)
            
    return gain

def ndcg(ours,ideal,k,mode=1):
    # ndcg@k
    dcgk = dcg(mode,ours,k)
    idcg = dcg(mode,ideal,k)
    return dcgk/idcg
        