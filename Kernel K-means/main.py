import numpy as np
from matplotlib import colors
import imageio
import os
from kmeans import kmeans

def rbf(u, v, g=10**-4):
    return np.exp(-1*g*kmeans.euclidean(kmeans, u, v))

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
    for i in range(len(gifs)):
        gif.append(imageio.get_reader('output/'+gifs[i]))

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

img_path = ['dataset/image1.png', 'dataset/image2.png']
gif_path = [['image1kernel k-means(k=2).gif','image1kernel k-means(k=3).gif','image1kernel k-means(k=4).gif','image1kernel k-means(k=5).gif','image1kernel k-means(k=6).gif'],
    ['image1kernel k-means(k=2, kmean++).gif','image1kernel k-means(k=3, kmean++).gif','image1kernel k-means(k=4, kmean++).gif','image1kernel k-means(k=5, kmean++).gif','image1kernel k-means(k=6, kmean++).gif'],
    ['image2kernel k-means(k=2).gif','image2kernel k-means(k=3).gif','image2kernel k-means(k=4).gif','image2kernel k-means(k=5).gif','image2kernel k-means(k=6).gif'], 
    ['image2kernel k-means(k=2, kmean++).gif','image2kernel k-means(k=3, kmean++).gif','image2kernel k-means(k=4, kmean++).gif','image2kernel k-means(k=5, kmean++).gif','image2kernel k-means(k=6, kmean++).gif']]

if not os.path.exists('./output'):
    os.mkdir('./output')

for i in range(2):
    print('processing image{}...'.format(i+1))
    img = imageio.imread(img_path[i])
    spatial_data, color_data = img_formater(img)
    gram = rbf(spatial_data, spatial_data) * rbf(color_data, color_data)

    max_fram = 0
    for j in range(5):
        km1 = kmeans(gram, k=2+j, method='default', is_kernel=True, keep_log=True)
        km2 = kmeans(gram, k=2+j, method='kmeans++', is_kernel=True, keep_log=True)

        record_trad, iter_trad = km1.run()
        visualizer(record_trad, 'output/'+gif_path[(i*2)][j])
        record_kpp, iter_kpp = km2.run()
        visualizer(record_kpp, 'output/'+gif_path[(i*2)+1][j])

        max_fram = max(max_fram, max(iter_kpp, iter_trad))
        if iter_kpp <= iter_trad:
            print('faster.....................[\033[94mkmeans++\033[0m]')
        else:
            print('faster.....................[\033[95mtradition\033[0m]')
        
    merge_gifs(gif_path[(i*2)], max_fram, i+1, 'default')
    merge_gifs(gif_path[(i*2)+1], max_fram, i+1, 'kpp')
