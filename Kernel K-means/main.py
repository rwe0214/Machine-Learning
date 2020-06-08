import numpy as np
import imageio
from matplotlib import colors
from matplotlib import pyplot as plt
import pandas as pd
import os

def euclidean(u, v):
    return np.matmul(u**2, np.ones((u.shape[1],v.shape[0]))) \
        -2*np.matmul(u, v.T) \
        +np.matmul(np.ones((u.shape[0], v.shape[1])), (v.T)**2)

def rbf(u, v, g=0.0001):
    return np.exp(-1*g*euclidean(u, v))

def img_formater(img):
    n = img.shape[0]*img.shape[1]
    spatial_data = []
    color_data = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            spatial_data.append([i, j])
            color_data.append(img[i][j])
    return np.array(spatial_data), np.array(color_data, dtype=int)

def get_distance(gram, ck):
    c_count = np.sum(ck, axis=0)
    dis = -2*np.matmul(gram, ck)/c_count + \
        np.matmul(np.ones(ck.shape), (np.matmul(ck.T, np.matmul(gram, ck)))*np.eye(ck.shape[1]))/(c_count**2)
    return dis

def kernel_k_means(gram, k=2, max_iter=100):
    #initial clusters
    n = gram.shape[0]
    ck = np.zeros((n, k))
    ck[np.arange(n),np.random.randint(k,size=n)] = 1
    
    record = []
    record.append(ck)
    for r in range(max_iter):
        print('running kernel k-means (k = {0}).........[{1}/{2}]'.format(k, r, max_iter), end='\r')
        #E-step
        dis = get_distance(gram, ck)

        #M-step
        update_ck = np.zeros(dis.shape)
        update_ck[np.arange(dis.shape[0]),np.argmin(dis, axis=1)] = 1
        delta_ck = np.count_nonzero(np.abs(update_ck - ck))

        record.append(update_ck)
        ck = update_ck
    print('running kernel k-means (k = {}).........[\033[92mcomplete\033[0m]'.format(k))
    return record

k_visual = colors.to_rgba_array(['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
def visualizer(record, save_path, k=2, figsize=(100,100,4)):
    print('visualizing.........', end='\r')
    gif = []
    for i in range(len(record)):
        c_id = np.argmax(record[i], axis=1)
        img = np.zeros(figsize, dtype=np.uint8)
        for j in range(c_id.shape[0]):
            m, n = (int(j/100), int(j%100))
            img[m][n] = 255*k_visual[c_id[j]]
        gif.append(img)
    imageio.mimsave(save_path, gif)
    print('visualizing.........[\033[92mcomplete\033[0m]')
        
def merge_gifs(gifs, id):
    #Create reader object for the gif
    gif = []
    for i in range(len(gifs)):
        gif.append(imageio.get_reader('output/'+gifs[i]))

    #Create writer object
    new_gif = imageio.get_writer('output/image'+str(id)+'.gif')

    for frame_number in range(100):
        img = []
        for i in range(len(gif)):
            img.append(gif[i].get_next_data())
        #here is the magic
        new_image = np.hstack(img)
        new_gif.append_data(new_image)
    for i in range(len(gif)):
        gif[i].close()
    new_gif.close()


img_path = ['dataset/image1.png', 'dataset/image2.png']
gif_path = [['image1kernel k-means(k=2).gif','image1kernel k-means(k=3).gif','image1kernel k-means(k=4).gif','image1kernel k-means(k=5).gif','image1kernel k-means(k=6).gif'],
    ['image2kernel k-means(k=2).gif','image2kernel k-means(k=3).gif','image2kernel k-means(k=4).gif','image2kernel k-means(k=5).gif','image2kernel k-means(k=6).gif']]


if not os.path.exists('./output'):
    os.mkdir('./output')
for i in range(2):
    print('processing image{}...'.format(i+1))
    img = imageio.imread(img_path[i])
    spatial_data, color_data = img_formater(img)
    gram = rbf(spatial_data, spatial_data) * rbf(color_data, color_data)
    
    for j in range(5):
        record = kernel_k_means(gram, k=2+j)
        visualizer(record, 'output/'+gif_path[i][j], k=2+i)
    merge_gifs(gif_path[i], i+1)
