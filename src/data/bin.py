import numpy as np

def logbin(count, x, scale=1.):
    xmax = x[-1]
    jmax = np.ceil(np.log(xmax) / np.log(scale))
    binedges = scale ** np.arange(jmax + 1)
    binedges = np.unique(binedges.astype('int64'))
    bin_centers = (binedges[:-1] * (binedges[1:])) ** 0.5
    bin_counts = np.zeros_like(bin_centers)
    count_filled = np.zeros(x[-1]+1)
    count_filled[x] = count
    count_filled = count_filled.astype('float')
    for i in range(len(bin_counts)):
        bin_counts[i] = np.sum(count_filled[binedges[i]:binedges[i+1]] / (binedges[i+1] - binedges[i] + 1))
    return bin_centers, bin_counts
