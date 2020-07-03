import numpy as np
import os
import PIL.Image
from matplotlib import pyplot as plt


def read_faces(dir_path):
    faces = []
    labels = []
    for filename in os.listdir(dir_path):
        with PIL.Image.open(dir_path + filename) as im:
            im = im.resize((100, 100), PIL.Image.BILINEAR)
            faces.append([np.array(im).reshape(1, -1)])
            labels.append(filename.split('.', 1)[0])
    faces = np.concatenate(faces, axis=0)
    faces = faces.reshape(faces.shape[0], faces.shape[2])
    faces = faces.astype('int64')
    return faces, labels


def show_faces(faces, filename=None, col=5):
    #f = faces.reshape(-1,231,195)
    f = faces.reshape(-1, 100, 100)
    n = f.shape[0]
    all_faces = []
    for i in range(int(n / col)):
        all_faces.append([np.concatenate(f[col * i:col * (i + 1)], axis=1)])

    all_faces = np.concatenate(all_faces[:], axis=1)
    all_faces = all_faces.reshape(all_faces.shape[1], all_faces.shape[2])

    plt.figure(figsize=(1.5 * col, 1.5 * n / col))
    plt.title(filename)
    plt.imshow(all_faces, cmap='gray')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15)

    if (filename):
        plt.savefig('./output/' + filename)
    else:
        plt.show()


def euclidean(u, v):
    return np.matmul(u**2, np.ones(
        (u.shape[1], v.shape[0]))) - 2 * np.matmul(u, v.T) + np.matmul(
            np.ones((u.shape[0], v.shape[1])), (v.T)**2)


def rbf(u, v, g=10**-11):
    return np.exp(-1 * g * euclidean(u, v))


def linear(u, v):
    return np.matmul(u, v.T)


def polynomial(u, v, g=0.7, coef0=10, d=5):
    return ((g * np.matmul(u, v.T)) + coef0)**d
