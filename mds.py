from sklearn.manifold import MDS


def get_absolute_values(distance_matrix):
    mds = MDS(dissimilarity='precomputed', n_components=1)
    y = mds.fit_transform(distance_matrix)
    y = [a[0] for a in y]
    return y
