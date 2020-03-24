#include "mnist.h"

#ifndef BIG_ENDIAN
#define BIG_ENDIAN 0
#endif
#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN 1
#endif

#define LEN_LABEL_INFO 2
#define LEN_IMAGE_INFO 4

#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define SIZE 784  // 28*28

#define PATH_TRAIN_LABEL "./mnist/train-labels-idx1-ubyte"
#define PATH_TRAIN_IMAGE "./mnist/train-images-idx3-ubyte"
#define PATH_TEST_LABEL "./mnist/t10k-labels-idx1-ubyte"
#define PATH_TEST_IMAGE "./mnist/t10k-images-idx3-ubyte"

int train_label_info[LEN_LABEL_INFO];
int train_image_info[LEN_IMAGE_INFO];
int test_label_info[LEN_LABEL_INFO];
int test_image_info[LEN_IMAGE_INFO];

uint8_t train_label[NUM_TRAIN][1];
uint8_t train_image[NUM_TRAIN][SIZE];
uint8_t test_label[NUM_TEST][1];
uint8_t test_image[NUM_TEST][SIZE];

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
        if (len_info == 2) {
            printf("%c\n", data[i][0] + 48);
        } else {
            for (int j = 0; j < size; j++) {
                if (j % 28 == 0)
                    printf("\n");
                if (data[i][j] >= 0 && data[i][j] <= 128)
                    printf("0 ");
                else
                    printf("1 ");
            }
        }
    }
}

void load_mnist()
{
    read_mnist(PATH_TRAIN_LABEL, LEN_LABEL_INFO, train_label_info, NUM_TRAIN, 1,
               train_label);
    read_mnist(PATH_TRAIN_IMAGE, LEN_IMAGE_INFO, train_image_info, NUM_TRAIN,
               SIZE, train_image);
    read_mnist(PATH_TEST_LABEL, LEN_LABEL_INFO, test_label_info, NUM_TEST, 1,
               test_label);
    read_mnist(PATH_TEST_IMAGE, LEN_IMAGE_INFO, test_image_info, NUM_TEST, SIZE,
               test_image);
}
