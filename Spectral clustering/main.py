import numpy as np
from spectral_clustering import spectral_clustering
from matplotlib import pyplot as plt
from matplotlib import colors
import imageio
import os

def euclidean(u, v):
        return np.matmul(u**2, np.ones((u.shape[1],v.shape[0]))) \
        -2*np.matmul(u, v.T) \
        +np.matmul(np.ones((u.shape[0], v.shape[1])), (v.T)**2)

def rbf(u, v, g=10**-4):
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

k_visual = colors.to_rgba_array(['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
def visualizer(record, save_path, figsize=(100,100,4)):
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
        
def merge_gifs(gifs, max_fram, id, method):
    gif = []
    for i in range(1, len(gifs)):
        gif.append(imageio.get_reader('output/'+gifs[i]+'.gif'))

    new_gif = imageio.get_writer('output/image'+str(id)+'_'+method+'.gif')
    
    if max_fram + 5 < 100:
        max_fram += 5
    for frame_number in range(max_fram):
        img = []
        for i in range(len(gif)):
            img.append(gif[i].get_next_data())
        new_image = np.hstack(img)
        new_gif.append_data(new_image)
    for i in range(len(gif)):
        gif[i].close()
    new_gif.close()

def reorder_by_cluster(c):
    new_order = np.array([])
    for i in range(c.shape[1]):
        new_order = np.append(new_order, np.where(c[:,i]==1)[0])
    new_order = new_order.astype('int32')
    return new_order

def show_vectors_by_clusters(vectors, clusters, fig_path):
    reorder_idx = reorder_by_cluster(clusters)
    num_cluster = np.sum(clusters, axis=0, dtype=int)

    iter_idx = 0
    for k in range(clusters.shape[1]):
        plt.subplot(1, clusters.shape[1], k+1)
        for j in range(num_cluster[k]):
            plt.plot(vectors[reorder_idx[iter_idx], :])
            plt.title('cluster '+str(k+1))
            iter_idx += 1
    plt.savefig(fig_path)
    plt.show(block = False)
    plt.pause(1)
    plt.close()

img_path = ['dataset/image1.png', 'dataset/image2.png']
gif_path = [['image1spectral clustering(k=3, normal)', 'image1spectral clustering(k=4, normal)', 'image1spectral clustering(k=5, normal)'], \
    ['image1spectral clustering(k=3, ratio)', 'image1spectral clustering(k=4, ratio)', 'image1spectral clustering(k=5, ratio)'], \
    ['image2spectral clustering(k=3, normal)', 'image2spectral clustering(k=4, normal)', 'image2spectral clustering(k=5, normal)'], \
    ['image2spectral clustering(k=3, ratio)', 'image2spectral clustering(k=4, ratio)', 'image2spectral clustering(k=5, ratio)']]

if not os.path.exists('./output'):
    os.mkdir('./output')

for i in range(2):
    print('processing image{}...'.format(i+1))
    img = imageio.imread(img_path[i])
    spatial_data, color_data = img_formater(img)
    similarity = rbf(spatial_data, spatial_data) * rbf(color_data, color_data)

    max_fram = 0
    for j in range(3):
        sc1 = spectral_clustering(similarity, k=3+j, normalize=True, keep_log=True)
        sc2 = spectral_clustering(similarity, k=3+j, normalize=False, keep_log=True)

        (record_norm, iter_norm), eigenvalues1, eigenvectors1 = sc1.run()
        visualizer(record_norm, 'output/'+gif_path[(i*2)][j]+'.gif')
        (record_ratio, iter_ratio), eigenvalues2, eigenvectors2 = sc2.run()
        visualizer(record_ratio, 'output/'+gif_path[(i*2)+1][j]+'.gif')

        max_fram = max(max_fram, max(iter_norm, iter_ratio))
        if iter_norm <= iter_ratio:
            print('faster.....................[\033[94mnormalize\033[0m]')
        else:
            print('faster.....................[\033[95munnormalize\033[0m]')
        show_vectors_by_clusters(eigenvectors1, record_norm[-1], 'output/'+gif_path[(i*2)][j]+'.png')
        show_vectors_by_clusters(eigenvectors2, record_ratio[-1], 'output/'+gif_path[(i*2)+1][j]+'.png')

    merge_gifs(gif_path[(i*2)], max_fram, i+1, 'normalize')
    merge_gifs(gif_path[(i*2)+1], max_fram, i+1, 'unnormalize')

