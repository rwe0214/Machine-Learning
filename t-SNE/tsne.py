#
# Modify from https://lvdmaaten.github.io/tsne/
# Make it has symmetric SNE and t-SNE at the same time

import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
from celluloid import Camera
import matplotlib.animation as animation


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def gradient_descend(iter, Y, iY, dY, gains, min_gain, momentum, eta):
    gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * (
        (dY > 0.) == (iY > 0.))
    gains[gains < min_gain] = min_gain
    iY = momentum * iY - eta * (gains * dY)
    Y = Y + iY
    Y = Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))

    return Y, iY, gains


def tsne(
        X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0,
        sym_sne=False):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01

    record = []
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    if sym_sne:
        record_ssne = []
        Y_ssne = Y.copy()
        dY_ssne = np.zeros((n, no_dims))
        iY_ssne = np.zeros((n, no_dims))
        gains_ssne = np.ones((n, no_dims))
        Q_ssne = np.zeros((n, n))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        if sym_sne:
            sum_Y_ssne = np.sum(np.square(Y_ssne), 1)
            num_ssne = -2. * np.dot(Y_ssne, Y_ssne.T)
            # different between t-sne region
            num_ssne = np.add(np.add(num_ssne, sum_Y_ssne).T, sum_Y_ssne)
            num_ssne = np.exp(-1 * num_ssne)
            # different between t-sne region
            num_ssne[range(n), range(n)] = 0.
            Q_ssne = num_ssne / np.sum(num_ssne)
            Q_ssne = np.maximum(Q_ssne, 1e-12)

        # Compute gradient
        PQ = P - Q
        if sym_sne:
            PQ_ssne = P - Q_ssne
        for i in range(n):
            dY[i, :] = np.sum(
                np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y),
                0)
            if sym_sne:
                # different between t-sne
                dY_ssne[i, :] = np.sum(
                    np.tile(PQ_ssne[:, i],
                            (no_dims, 1)).T * (Y_ssne[i, :] - Y_ssne), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        Y, iY, gains = gradient_descend(iter, Y, iY, dY, gains, min_gain,
                                        momentum, eta)
        if sym_sne:
            Y_ssne, iY_ssne, gains_ssne = gradient_descend(
                iter, Y_ssne, iY_ssne, dY_ssne, gains_ssne, min_gain, momentum,
                eta)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            record.append(Y)
            if sym_sne:
                C_ssne = np.sum(P * np.log(P / Q_ssne))
                record_ssne.append(Y_ssne)
                print("Iter %4d: tsne error = %f, sym-sne error = %f" %
                      (iter + 1, C, C_ssne))
            else:
                print("Iter %4d: tsne error = %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    if not sym_sne:
        return record, P, Q
    else:
        return (record, record_ssne), P, (Q, Q_ssne)


def make_gif(record, labels, method, perplexity):
    camera = Camera(plt.figure())
    plt.title(method + ' with Perplexity=' + str(perplexity))
    for i in range(len(record)):
        img = plt.scatter(record[i][:, 0], record[i][:, 1], 20, labels)
        camera.snap()
    anim = camera.animate(interval=5, repeat_delay=20)
    anim.save(
        'output/' + method + '_' + str(perplexity) + '.gif', writer='pillow')
    plt.scatter(record[-1][:, 0], record[-1][:, 1], 20, labels)
    plt.savefig('output/' + method + '_' + str(perplexity) + '.png')


def show_similarity(S, labels, title, filename, perplexity):
    n = len(S)
    sort_idx = np.concatenate(
        [np.where(labels == l)[0] for l in np.unique(labels)])
    plt.figure(figsize=(10 * n, 7.5))
    S = [np.log(p[:, sort_idx][sort_idx, :]) for p in S]
    all_min = min([np.min(p) for p in S])
    all_max = max([np.max(p) for p in S])
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.title(title[i])
        im = plt.imshow(S[i], cmap='gray', vmin=all_min, vmax=all_max)
        plt.colorbar(im)
    plt.savefig('output/' + filename + '_' + str(perplexity) + '.png')


if __name__ == "__main__":
    print(
        "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
    )
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    if not os.path.exists('./output'):
        os.mkdir('./output')

    perplexity = 2**np.arange(2, 13, 2, dtype=float)

    X = pca(X, 50).real
    for per in perplexity:
        (Y, Y_ssne), P, (Q, Q_ssne) = tsne(X, 2, 50, pre, sym_sne=True)

        make_gif(Y, labels, 't-sne', pre)
        make_gif(Y_ssne, labels, 'sym-sne', pre)
        show_similarity(
            list([P, Q, Q_ssne]), labels,
            list(['P', 'Q from t-SNE', 'Q from sym-SNE']), 'similarity',
            pre)
