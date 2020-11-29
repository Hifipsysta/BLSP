
import math
import time
import numpy as np
from PIL import Image
import matplotlib.image as mp
import matplotlib.pyplot as plt


img = Image.open('lena_RGB.jpg')
im = img.convert('L')
#im=img[:,:,0]
plt.imshow(im,cmap='gray')
plt.axis('off')
plt.show()

print(np.array(im).shape)

rsize=np.array(im).shape[0]
csize=np.array(im).shape[1]


#生成高斯随机测量矩阵
sampleRate=0.0625  #采样率
Phi=np.random.randn(int(rsize*sampleRate),csize)
# Phi=np.random.randn(256,256)
# u, s, vh = np.linalg.svd(Phi)
# Phi = u[:256*sampleRate,] #将测量矩阵正交化



#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((rsize,csize))
v=range(csize)
for k in range(0,csize):
    dct_1d=np.cos(np.dot(v,k*math.pi/csize))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

#随机测量
img_cs_1d=np.dot(Phi,im)

#CoSaMP算法函数
def cs_CoSaMP(y,D):
    S=math.floor(y.shape[0]/4)  #稀疏度
    residual=y  #初始化残差
    pos_last=np.array([],dtype=np.int64)
    result=np.zeros((csize))

    for j in range(S):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos_temp=np.argsort(product)
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos_temp=pos_temp[0:2*S]#对应步骤3
        pos=np.union1d(pos_temp,pos_last)

        result_temp=np.zeros((csize))
        result_temp[pos]=np.dot(np.linalg.pinv(D[:,pos]),y)

        pos_temp=np.argsort(np.fabs(result_temp))
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        result[pos_temp[:S]]=result_temp[pos_temp[:S]]
        pos_last=pos_temp
        residual=y-np.dot(D,result)
    return  result


sparse_rec_1d=np.zeros((rsize,csize))   # 初始化稀疏系数矩阵
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵

start=time.time()
for i in range(csize):
    print('正在重建第',i,'列...')
    column_rec=cs_CoSaMP(img_cs_1d[:,i],Theta_1d)  #利用CoSaMP算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec;
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

end=time.time()
time_consume = end - start
print(time_consume)

'''
#均值池化
def avg_pooling(data, m, n):
    a,b = data.shape
    img_new = []
    for i in range(0,a,m):
        line = []#记录每一行
        for j in range(0,b,n):
            x = data[i:i+m,j:j+n]#选取池化区域
            line.append(np.sum(x)/(n*m))
        img_new.append(line)
    return np.array(img_new)
'''


#显示重建后的图片
image2=Image.fromarray(img_rec)
plt.imshow(image2,cmap='gray')
mp.imsave('lena_CoSoMP0.0625.jpg',image2,cmap='gray')

error = np.linalg.norm(img_rec-im)/np.linalg.norm(im)
print('重构准确率',1-error)








