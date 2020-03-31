#ifndef __MNIST_H
#define __MNIST_H

#ifndef BIG_ENDIAN
#define BIG_ENDIAN 0
#endif
#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN 1
#endif

#define LEN_LABEL_INFO 2
#define LEN_IMAGE_INFO 4

#define NUM_MNIST_TRAIN 60000
#define NUM_MNIST_TEST 10000
#define SIZE_MNIST 784  // 28*28

#define PATH_MNISTTRAIN_LABEL "./dataset/mnist/train-labels-idx1-ubyte"
#define PATH_MNISTTRAIN_IMAGE "./dataset/mnist/train-images-idx3-ubyte"
#define PATH_MNISTTEST_LABEL "./dataset/mnist/t10k-labels-idx1-ubyte"
#define PATH_MNISTTEST_IMAGE "./dataset/mnist/t10k-images-idx3-ubyte"

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

typedef struct MNIST_INFO {
    int train_size;
    int test_size;
    int label_class_size;
    int image_size;
} mnist_info;

typedef struct MNIST_DATA {
    /*uint8_t train_image[NUM_MNIST_TRAIN][SIZE_MNIST];
    uint8_t train_label[NUM_MNIST_TRAIN][1];
    uint8_t test_image[NUM_MNIST_TEST][SIZE_MNIST];
    uint8_t test_label[NUM_MNIST_TEST][1];*/
    uint8_t **train_image;
    uint8_t *train_label;
    uint8_t **test_image;
    uint8_t *test_label;
} mnist_data;

int is_little_endian();
void read_or_fail(int fd, void *usr_buf, size_t n);
ssize_t rio_readn(int fd, void *usr_buf, size_t n);
int open_or_fail(char *path, int flag);
void swap(unsigned char *b1, unsigned char *b2);
void swap_bytes(unsigned char *ptr, size_t len);
void statistic_mnist_train_data();
void read_mnist(char *path,
                int len_info,
                int data_info[],
                int num_data,
                int size,
                uint8_t data[][size]);
void load_mnist();
mnist_info *get_mnist_info();
mnist_data *get_mnist_data();
void free_mnist_info(mnist_info *info);
void free_mnist_data(mnist_data *data);

#endif
