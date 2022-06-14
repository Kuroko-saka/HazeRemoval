import cv2
from PIL import Image
import numpy as np
import os

def read_img(imageName):
    cur_dir = '/'.join(os.path.abspath(__file__).split('\\')[:-1])
    imgfile = cur_dir + imageName
    img = cv2.imread(imgfile,1)
    # 将图像通道分离开。
    b, g, r = cv2.split(img)
    # 以RGB的形式重新组合。
    rgb_image = cv2.merge([r, g, b])
    return rgb_image

def show_img(imageName):
    img = Image.fromarray(imageName)
    # img.save("36-result.png")
    img.show()
    return img
    
def get_dark(img):
    darkImage = img.min(axis=2)
    return darkImage

def get_Jdark(dark,z=15):
    row = dark.shape[0]
    loc = dark.shape[1]
    Jdark=np.zeros(dark.shape,dark.dtype)
    for i in range(row):
        for j in range(loc):
            min_value = np.min(img[max(0,i-z):min(row,i+z+1),max(0,j-z):min(loc,j+z+1)])
            Jdark[i][j] = min_value
    Jdark = np.uint8(Jdark)
    return Jdark

def get_A(img,Jdark):
    row = Jdark.shape[0]
    loc = Jdark.shape[1]
    total_count = int(row * loc * 0.001)
    img_c = img.copy()
    Jdark_c = Jdark.copy()
    img_c = img_c.reshape(-1,3)
    Jdark_c = Jdark_c.reshape(-1)
    index = Jdark_c.argsort()[::-1]
    index = index[0:total_count]
    A = img_c[index]
    A0= max(A[:,0])/255
    A1 = max(A[:,1])/255
    A2 = max(A[:,2])/255
    A = np.array([A0,A1,A2])
    return A

def get_t(img,A,w=0.95):
    img_t = np.zeros(img.shape,img.dtype)
    for c in range(0,3):
        img_t[:,:,c] = img[:,:,c]/A[c]
    t_dark = get_dark(img_t)
    t_Jdark = get_Jdark(t_dark,15)
    t = 1 - w*t_Jdark/255
    return t

def get_J(img,A,t_):
    img = img/255
    J = np.zeros(img.shape,img.dtype)
    t_ = np.where((t_ >= 0.1) , t_, 0.1)
    for c in range(0,3):
        J[:,:,c] = (img[:,:,c]-A[c])/t_ + A[c]

    return J*255


def Guide_filter(img, t, r=60,eps = 0.0001):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = np.float64(img_gray)/255
    
    mean_I = cv2.boxFilter(img_gray,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(t, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(img_gray*t,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(img_gray*img_gray,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a * img_gray + mean_b
    return q

if __name__ =="__main__":
    imageName = "/litt.jpg"
    # imageName = "/36_hazy.png"

    img = read_img(imageName)
    print("读文件完成")
    dark = get_dark(img)
    print("获取暗通道完成")
    Jdark = get_Jdark(dark,15)
    print("暗通道最小值滤波完成")
    A = get_A(img,Jdark)
    print("计算A完成")
    t = get_t(img,A,0.95)
    print("计算t完成")
    t_ = Guide_filter(img,t,60,0.0001)
    print("引导滤波完成")
    J = get_J(np.float64(img),A,t_)
    print("计算J完成")
    res = np.hstack((img,J))
    show_img(np.uint8(res))

    
    

