import os
import math
import time
import numpy as np
from PIL import Image
import matplotlib.image as mp
import matplotlib.pyplot as plt
from torchvision.utils import save_image


img = Image.open('lena_RGB.jpg')
im = img.convert('L')
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.show()
mp.imsave('lena_gray.jpg',im,cmap='gray')


#池化
def pooling(data, Wp, Hp, M, N):
    W,H = data.shape    #获得图片尺寸
    img_new = []
    for i in range(0,W,Wp):   #从0开始到W结束，间隔为池化核的宽
        line = []#记录每一行
        for j in range(0,H,Hp):     #从0开始到H结束，间隔为池化核的高
            x = data[i:i+Wp,j:j+Hp]     #选取池化区域：宽度为池化核的宽，高度为池化核的高
            line.append(x[M,N])      #我到底只要哪个有序位置上的像素
        img_new.append(line)         #合成进去形成列表
    return np.array(img_new)

def inverse_LSP(pooled_array,W,H):
    X = np.zeros((W, H))
    M,N,Wp,Hp = pooled_array.shape
    for m in range(M):
        for n in range(N):
            pooled_img = pooled_array[m,n]
            for i in range(Wp): #128
                for j in range(Hp): #128
                    X[m+M*i,n+N*j]=pooled_img[i,j]
    return X

'''
def inverse_pooling(pooled_array,W,H,M,N):
    #X = np.zeros((W, H))
    Q = len(pooled_list)
    list_=[]
    Wp, Hp = pooled_array[0,0].shape
    for i in range(Wp):  # i,j代表一个池化区域的什么位置    128次
        for j in range(Hp):  # 128次
            area_=[]
            for q in range(Q):   #16次
                pooled_img = pooled_list[q]  # 每张图片就是池化区域
                area_.append(pooled_img[i,j])
            array_area=np.array(area_)
            array_area = array_area.reshape((4,4))
            list_.append(array_area)
    return list_
            #array_ = np.array(list_)
            #array_ = array_.reshape((M,N))
'''

def shift_pooling(data,M,N):
    img_list=[]
    for i in range(M):
        for j in range(N):
            picture_ = pooling(data,M,N,i,j)
            img_list.append(picture_)
    return img_list

M=4
N=4
Wp=128
Hp=128


start=time.time()
pooled_list = shift_pooling(np.array(im), M, N)  #2,2; 4,4
end1=time.time()
time_consume1 = end1 - start
pooled_array = np.array(pooled_list)
pooled_array = pooled_array.reshape((M,N,Wp,Hp)) #2,2,256,256; 4,4,128,128
#print(pooled_array.shape)


X = inverse_LSP(pooled_array,512,512)
end2=time.time()
print(X.shape)
plt.imshow(X,cmap='gray')
plt.axis('off')
plt.show()
mp.imsave('ILSP_generate.jpg',X,cmap='gray')

error = np.linalg.norm(X-im)/np.linalg.norm(im)
print('重构准确率',1-error)



#print(pooled_array.shape)
#list_ = inverse_pooling(pooled_list,512,512,4,4)
#print(len(list_))
#X = inverse_shift_pooling(list_,512,512,4,4)

#X = scamming(pooled_list,128,128)


#print(pooled_list[0].shape)
#print(type(X))
'''
#print(X.shape)
plt.imshow(X,cmap='gray')
plt.axis('off')
plt.show()
'''

time_consume2 = end2 - start


sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

print('生成图片数量：',len(pooled_list))
print('LSP时间成本：',time_consume1)
print('ILSP时间成本',time_consume2-time_consume1)
print('BLSP时间成本：',time_consume2)


'''
for i in range(M):   #2;  4
    for j in range(N):    #2 ; 4
        plt.imshow(pooled_array[i,j],cmap='gray')
        plt.axis('off')
        plt.show()
'''


for i in range(len(pooled_list)):
    plt.imshow(pooled_list[i],cmap='gray')
    plt.axis('off')
    plt.show()
    mp.imsave(os.path.join(sample_dir, 'lena_pooled{}.jpg'.format(i + 1)),pooled_list[i],cmap='gray')




