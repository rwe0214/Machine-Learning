import numpy as np
from util import *
from PCA import PCA
from LDA import LDA
from KNN import KNN

DATASET_DIR = './Yale_Face_Database/'
TRAIN_DIR = DATASET_DIR + 'Training/'
TEST_DIR = DATASET_DIR + 'Testing/'
if not os.path.exists('./output'):
    os.mkdir('./output')

train_faces, train_labels = read_faces(TRAIN_DIR)
test_faces, test_labels = read_faces(TEST_DIR)

#PCA
print('PCA')
pca = PCA(train_faces, k=25)
pca_space, W = pca.run()

random_faces = np.vstack([
    train_faces[r]
    for r in np.random.randint(0, train_faces.shape[0], size=10)
])
reconstruct_faces = np.matmul(np.matmul(random_faces, W), W.T)

show_faces(random_faces, '10_random_faces.png')
show_faces(W.T, 'PCA_25_eigenfaces.png')
show_faces(reconstruct_faces, 'PCA_10_reconstruct_faces.png')

pca = PCA(train_faces, k=25)
train_space, W_train = pca.run()
test_space = np.matmul(test_faces, W_train)

knn = KNN(train_space, test_space, train_labels, k=5)
predict = knn.run()

correct = 0
for i in range(len(predict)):
    if predict[i] == test_labels[i]:
        correct += 1
print('Face-reconition accuracy: {}/{} = {:.2f}%'.format(
    correct, len(predict), correct / len(predict) * 100.00))

#kernel PCA
print('\nKernel PCA')
kernel_info = [rbf, polynomial, linear]

for idx, kernel in enumerate(kernel_info):
    K_train = kernel(train_faces, train_faces)
    K_test_train = kernel(test_faces, train_faces)

    kpca = PCA(K_train, k=25, is_kernel=True)
    train_space, alpha = kpca.run()

    NM1 = np.ones(K_test_train.shape) / K_train.shape[0]
    test_space = np.matmul(K_test_train - np.matmul(NM1, K_train), alpha)

    knn = KNN(train_space, test_space, train_labels, k=5)
    predict = knn.run()
    correct = 0
    for i in range(len(predict)):
        if predict[i] == test_labels[i]:
            correct += 1
    print('({}) Face-reconition accuracy: {}/{} = {:.2f}%'.format(
        kernel.__name__, correct, len(predict),
        correct / len(predict) * 100.00))

#LDA
print('\nLDA')
train_faces = np.resize(train_faces, (train_faces.shape[0], 10000))
lda = LDA(train_faces, np.array(train_labels), k=25)
train_space, W_train = lda.run()

reconstruct_faces = np.matmul(np.matmul(random_faces, W_train), W_train.T)

show_faces(W_train.T, 'LDA_25_eigenfaces.png')
show_faces(reconstruct_faces, 'LDA_10_reconstruct_faces.png')

test_space = np.matmul(test_faces, W_train)

knn = KNN(train_space, test_space, train_labels, k=5)
predict = knn.run()

correct = 0
for i in range(len(predict)):
    if predict[i] == test_labels[i]:
        correct += 1
print('Face-reconition accuracy: {}/{} = {:.2f}%'.format(
    correct, len(predict), correct / len(predict) * 100.00))

#kernel LDA
print('\nKernel LDA')
kernel_info = [rbf, polynomial, linear]

for idx, kernel in enumerate(kernel_info):
    K_train = kernel(train_faces, train_faces)
    K_test_train = kernel(test_faces, train_faces)

    klda = LDA(K_train, np.array(train_labels), k=25, is_kernel=True)
    train_space, alpha = klda.run()

    test_space = np.matmul(K_test_train, alpha)

    knn = KNN(train_space, test_space, train_labels, k=5)
    predict = knn.run()
    correct = 0
    for i in range(len(predict)):
        if predict[i] == test_labels[i]:
            correct += 1
    print('({}) Face-reconition accuracy: {}/{} = {:.2f}%'.format(
        kernel.__name__, correct, len(predict),
        correct / len(predict) * 100.00))
