#ifndef __MNIST_H
#define __MNIST_H
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int is_little_endian();
void read_or_fail(int fd, void *usr_buf, size_t n);
ssize_t rio_readn(int fd, void *usr_buf, size_t n);
int open_or_fail(char *path, int flag);
void swap(unsigned char *b1, unsigned char *b2);
void swap_bytes(unsigned char *ptr, size_t len);
void read_mnist(char *path,
                int len_info,
                int data_info[],
                int num_data,
                int size,
                uint8_t data[][size]);
void load_mnist();

#endif
