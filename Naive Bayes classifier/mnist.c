#include "mnist.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int mnist_train_label_info[LEN_LABEL_INFO];
int mnist_train_image_info[LEN_IMAGE_INFO];
int mnist_test_label_info[LEN_LABEL_INFO];
int mnist_test_image_info[LEN_IMAGE_INFO];

double mnist_count_label[10];

uint8_t mnist_train_label[NUM_MNIST_TRAIN][1];
uint8_t mnist_train_image[NUM_MNIST_TRAIN][SIZE_MNIST];
uint8_t mnist_test_label[NUM_MNIST_TEST][1];
uint8_t mnist_test_image[NUM_MNIST_TEST][SIZE_MNIST];

double mnist_train_label_mean[10][SIZE_MNIST];
double mnist_train_label_varience[10][SIZE_MNIST];
int is_little_endian()
{
    uint16_t test = 0x0001;
    uint8_t *b = (uint8_t *) &test;
    return (*b) ? LITTLE_ENDIAN : BIG_ENDIAN;
}

void read_or_fail(int fd, void *usr_buf, size_t n)
{
    if (rio_readn(fd, usr_buf, n) < 0) {
        perror("read");
        fprintf(stderr, "read fail\n");
        exit(EXIT_FAILURE);
    }
}

ssize_t rio_readn(int fd, void *usr_buf, size_t n)
{
    size_t nleft = n;
    ssize_t nread;
    char *bufp = usr_buf;

    while (nleft > 0) {
        if ((nread = read(fd, bufp, nleft)) < 0) {
            if (errno == EINTR) /* interrupted by sig handler return */
                nread = 0;      /* and call read() again */
            else
                return -1; /* errno set by read() */
        } else if (nread == 0)
            break; /* EOF */
        nleft -= nread;
        bufp += nread;
    }
    return (n - nleft); /* return >= 0 */
}

int open_or_fail(char *path, int flag)
{
    int fd;
    if ((fd = open(path, flag)) == -1) {
        perror("open");
        fprintf(stderr, "Could open %s\n", path);
        exit(EXIT_FAILURE);
    }
    return fd;
}

void swap(uint8_t *b1, uint8_t *b2)
{
    *b1 ^= *b2;
    *b2 ^= *b1;
    *b1 ^= *b2;
}

void swap_bytes(uint8_t *ptr, size_t len)
{
    for (int i = 0; i < len / 2; i++)
        swap(ptr + i, ptr + len - 1 - i);
}

/*void uint8_t_to_double(int size_data, uint8_t data[][SIZE_MNIST]){
    for(int i=0; i<size_data; i++)
        for(int j=0; j<SIZE_MNIST; j++)
            mnist_train_data_double
}*/

void statistic_mnist_train_data()
{
    /*get mean*/
    for (int i = 0; i < NUM_MNIST_TRAIN; i++) {
        int label = mnist_train_label[i][0];
        for (int j = 0; j < SIZE_MNIST; j++) {
            mnist_train_label_mean[label][j] += mnist_train_image[i][j];
        }
        mnist_count_label[label]++;
    }



    for (int i = 0; i < 10; i++)
        for (int j = 0; j < SIZE_MNIST; j++)
            mnist_train_label_mean[i][j] /= mnist_count_label[i];

    /*get varience*/
    for (int i = 0; i < NUM_MNIST_TRAIN; i++) {
        int label = mnist_train_label[i][0];
        for (int j = 0; j < SIZE_MNIST; j++) {
            mnist_train_label_varience[label][j] += pow(
                mnist_train_image[i][j] - mnist_train_label_mean[label][j], 2);
        }
    }
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < SIZE_MNIST; j++) {
            mnist_train_label_varience[i][j] =
                sqrt(mnist_train_label_varience[i][j] / mnist_count_label[i]);
        }
}

void read_mnist(char *path,
                int len_info,
                int data_info[],
                int num_data,
                int size,
                uint8_t data[][size])
{
    int fd;
    uint8_t *ptr;
    fd = open_or_fail(path, O_RDONLY);

    read_or_fail(fd, data_info, len_info * sizeof(int));

    if (is_little_endian())
        for (int i = 0; i < len_info; i++) {
            ptr = (uint8_t *) (data_info + i);
            swap_bytes(ptr, sizeof(int));
            ptr += sizeof(int);
        }

    for (int i = 0; i < num_data; i++) {
        read_or_fail(fd, data[i], size * sizeof(uint8_t));
    }
}

int get_MNIST_train_size()
{
    return SIZE_MNIST;
}

void load_mnist()
{
    read_mnist(PATH_MNISTTRAIN_LABEL, LEN_LABEL_INFO, mnist_train_label_info,
               NUM_MNIST_TRAIN, 1, mnist_train_label);
    read_mnist(PATH_MNISTTRAIN_IMAGE, LEN_IMAGE_INFO, mnist_train_image_info,
               NUM_MNIST_TRAIN, SIZE_MNIST, mnist_train_image);
    read_mnist(PATH_MNISTTEST_LABEL, LEN_LABEL_INFO, mnist_test_label_info,
               NUM_MNIST_TEST, 1, mnist_test_label);
    read_mnist(PATH_MNISTTEST_IMAGE, LEN_IMAGE_INFO, mnist_test_image_info,
               NUM_MNIST_TEST, SIZE_MNIST, mnist_test_image);
    statistic_mnist_train_data();
}

mnist_info *get_mnist_info()
{
    mnist_info *info = malloc(sizeof(mnist_info));

    info->train_size = NUM_MNIST_TRAIN;
    info->test_size = NUM_MNIST_TEST;
    info->label_class_size = 10;
    info->image_size = 28 * 28;

    return info;
}

mnist_data *get_mnist_data()
{
    mnist_data *data = malloc(sizeof(mnist_data));
    for (int i = 0; i < NUM_MNIST_TRAIN; i++)
        for (int j = 0; j < SIZE_MNIST; j++)
            data->train_image[i][j] = mnist_train_image[i][j];
    for (int i = 0; i < NUM_MNIST_TRAIN; i++)
        data->train_label[i][0] = mnist_train_label[i][0];
    for (int i = 0; i < NUM_MNIST_TEST; i++)
        for (int j = 0; j < SIZE_MNIST; j++)
            data->test_image[i][j] = mnist_test_image[i][j];
    for (int i = 0; i < NUM_MNIST_TEST; i++)
        data->train_label[i][0] = mnist_train_label[i][0];

    return data;
}
