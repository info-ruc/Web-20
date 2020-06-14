from PIL import Image
import math
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from pylab import *
from skimage import transform

global count
count=0
aimpath=sys.argv[2]
def decomp_img(img_name):
    global count
    img=mpimg.imread(img_name)
    (x,y,_)=img.shape
    if x==y:
        mpimg.imsave(aimpath+"/"+str(count)+".jpg",img)
        count+=1
    elif y>x:
        if y%x==0:
            n_piece=y//x
            for i in range(n_piece):
                cut_img=img[i*x:(i+1)*x,:,:]
                cut_img=transform.resize(cut_img,(256,256))
                mpimg.imsave(aimpath+"/"+str(count)+".jpg",cut_img)
                count+=1
        else:
            n_piece=math.ceil(y/x)
            n_cover=math.floor(y/x)
            d_cover_length=(x*n_piece-y)//n_cover
            top=0
            bot=x
            for i in range(n_piece):
                cut_img=img[:,top:bot,:]
                cut_img=transform.resize(cut_img,(256,256))
                mpimg.imsave(aimpath+"/"+str(count)+".jpg",cut_img)
                count+=1
                top=bot-d_cover_length
                bot=top+x
    else:
        if x%y==0:
            n_piece=x//y
            for i in range(n_piece):
                cut_img=img[:,i*y:(i+1)*y,:]
                cut_img=transform.resize(cut_img,(256,256))
                mpimg.imsave(aimpath+"/"+str(count)+".jpg",cut_img)
                count+=1
        else:
            n_piece=math.ceil(x/y)
            n_cover=math.floor(x/y)
            d_cover_length=(y*n_piece-x)//n_cover
            #print(n_cover)
            top=0
            bot=y
            for i in range(n_piece):
                #print(img.shape)
                #print(top,bot)
                cut_img=img[top:bot,:,:]
                #print(cut_img.shape)
                cut_img=transform.resize(cut_img,(256,256))
                mpimg.imsave(aimpath+"/"+str(count)+".jpg",cut_img)
                count+=1
                top=bot-d_cover_length
                bot=top+y
path=sys.argv[1]
filelist=os.listdir(path)
for files in filelist:
    if files[0]==".":
        continue
    if os.path.isdir(os.path.join(path,files)):
        continue
    decomp_img(os.path.join(path,files))