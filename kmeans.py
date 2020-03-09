# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 22:00:21 2020

@author: madhu
"""

from PIL import Image
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


im=Image.open('face.jpg')
pixelMap=im.load()
imgnew=Image.new(im.mode,im.size)
pixelnew=imgnew.load()

k=16
c=[]
for i in range(k):
    c.append(pixelMap[random.randint(0,im.size[0]-1),random.randint(0,im.size[1]-1)])
print(c)

x=[]
itr=0
cluster=[]
while itr<1:
    #finding clusters
    t=c.copy()
    
    cluster=[]
    for i in range(k):
      cluster.append([])
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            dist=[]
            for m in range(k):
                dist.append(np.linalg.norm(np.array(c[m])-np.array(pixelMap[i,j])))
            cluster[dist.index(min(dist))].append(pixelMap[i,j])
            
    #finding centroid
    temp=[]
    for i in range(k):
        s=np.array([0,0,0])
        for j in cluster[i]:
            s[0]+=j[0]
            s[1]+=j[1]
            s[2]+=j[2]
        s=s/len(cluster[i])
        temp.append(s)
    for i in range(k):
        c[i]=temp[i]     
      
    itr+=1
    
final=[]
for i in range(k):
    final.append([])
    for j in range(3):
        final[i].append(int(math.floor(c[i][j])))
print(final)

for i in range(im.size[0]):
    for j in range(im.size[1]):
        dist=[]
        for m in range(k):
            dist.append(np.linalg.norm(np.array(final[m])-np.array(pixelMap[i,j])))
        pixelnew[i,j]=tuple(final[dist.index(min(dist))])
        
        
plt.subplot(1,2,1)
plt.imshow(imgnew)
plt.title('Compressed',color='w')
plt.subplot(1,2,2)
plt.imshow(im)
plt.title('Original',color='w')
plt.show()