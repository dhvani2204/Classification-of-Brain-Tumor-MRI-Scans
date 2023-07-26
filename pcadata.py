import numpy
import numpy as np
from sklearn.decomposition import PCA
import data


def pcaData():
    images = numpy.load('BINData.npy')
    masks = numpy.load('BINMasks.npy')
    PCAimg = []
    PCAmsk = []

    for img in images:
        pca = PCA(n_components=65)
        img_pca = pca.fit_transform(img)
        # img_inv = pca.inverse_transform(img_pca)
        PCAimg.append(img_pca)

    for msk in masks:
        pca = PCA(n_components=65)
        img_pca = pca.fit_transform(msk)
        # img_inv = pca.inverse_transform(img_pca)
        PCAmsk.append(img_pca)

    X_pca = np.asarray(PCAimg)
    X_pca_msk = np.asarray(PCAmsk)

    numpy.save('PCAData.npy', X_pca)
    numpy.save('PCAMasks.npy', X_pca_msk)


if __name__ == "__main__":
    pcaData()
