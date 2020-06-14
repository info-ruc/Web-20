from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import *
import random
import math
import time
import sys

record=[]
begin=int(sys.argv[2])
end=int(sys.argv[3])
print("需要从前两张图片中选择一个保留原图并迁移WLOP风格更好的图片\n觉得第一张好的输入1觉得第二张好的输入2\n第三张为原图，第四张为参考的WLOP风格")
for i in range(begin,end):
    zero=2-int(math.log10(i))
    imgname1="checkpoints/"+sys.argv[1]+"/web/images/epoch"+"0"*zero+str(i)+"_fake_B.png"
    imgname2="checkpoints/"+sys.argv[1]+"/NST/NST"+str(i)+".png"
    imgname3="checkpoints/"+sys.argv[1]+"/web/images/epoch"+"0"*zero+str(i)+"_real_A.png"
    imgname4="checkpoints/"+sys.argv[1]+"/web/images/epoch"+"0"*zero+str(i)+"_real_B.png"
    first=random.randint(0,1)
    im=[]
    im.append(Image.open(imgname1))
    im.append(Image.open(imgname2))
    im.append(Image.open(imgname3))
    im.append(Image.open(imgname4))
    plt.ion()
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(im[first])
    plt.subplot(1,4,2)
    plt.imshow(im[1-first])
    plt.show()
    plt.subplot(1,4,3)
    plt.imshow(im[2])
    plt.subplot(1,4,4)
    plt.imshow(im[3])
    plt.pause(5)
    plt.close()
    result=int(input())
    assert(result==1 or result==2)
    if first==0:
        record.append(result-1)
    else:
        record.append(2-result)
fl=open("result.txt",'w')
for i in record:
    fl.write(str(i)+"\n")
fl.close()