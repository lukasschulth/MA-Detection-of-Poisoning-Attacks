"""
kmeans mit Euklidischer Distanz

"""


import os

from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.random import choice
from PIL import Image
import ot
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import distance_matrix
from scipy.sparse.linalg import eigs, eigsh
import sys
from sklearn.metrics import confusion_matrix
import random


def heatmap_to_rel_coord(im):
        """Transform Relevance map to list of coordinates and relevances per pixel in the range of [0,1]

        Input:
        -------
                    im: 2D image with one channel

        Returns:
        -----------
        xy = list of tuples (x,y)
        r = list of mass per point
        """

        x = []
        y = []
        r = []
        xy = []
        for i in range(32):
            for j in range(32):
                x.append(i)
                y.append(j)
                r.append(im[i][j])
                xy.append([i, j])
        xy = np.asarray(xy)
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.asarray(r)

        # Normalize "by hand"
        max = r.max()
        min = r.min()
        r = (r-min)/(max-min)

        # Division by total mass
        r = r/r.sum()

        return xy, x, y, r


def compute_GWD_matrix(heatmap_array, method='GWD', verbose=False):

        # Initialize matrix:
        GWD = np.zeros((heatmap_array.shape[0], heatmap_array.shape[0]))
        # Iterate through matrix(we only need upper oder lower triangle since distances are symmetric
        for i in tqdm(range(heatmap_array.shape[0])):
            for j in range(i-1):
                im1 = heatmap_array[i]
                im2 = heatmap_array[j]
                xy1, x1, y1, r1 = heatmap_to_rel_coord(im1)
                xy2, x2, y2, r2 = heatmap_to_rel_coord(im2)

                C1 = sp.spatial.distance.cdist(xy1, xy1)
                C2 = sp.spatial.distance.cdist(xy2, xy2)

                C1 /= C1.max()
                C2 /= C2.max()

                if method == 'GWD':
                    gw, log = ot.gromov.gromov_wasserstein(
                    C1, C2, r1, r2, 'square_loss', verbose=verbose, log=True)
                elif method == 'EGWD':
                    gw, log = ot.gromov.entropic_gromov_wasserstein(
                    C1, C2, r1, r2, 'square_loss', epsilon=5e-4, log=True, verbose=verbose)
                else:
                    print('METHOD not implemented.')

                GWD[i][j] = log['gw_dist']

                #Use symmetry
                GWD[j][i] = GWD[i][j]


            # Setze Diagonalelement =0
            GWD[i][i] = 0

        return GWD


if __name__ == '__main__':
    y_true = [0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1]

    print(1- np.asarray(y_true))
    a,b,c,d = confusion_matrix(y_true, y_pred).ravel()
    #a,b,c,d = confusion_matrix(poison_labels, clustering_result).ravel()
    #specificity = tn / (tn+fp)
    #print(tn, fp, fn, tp)
    print(a,b,c,d)

    #sys.exit()
    # Open images of suspicious class
    #path = '/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/LRP_Outputs/incv3_matthias_v2e-rule/relevances/00026/'
    path = '/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/LRP_Outputs/incv3_20_epochs_normalizede-rule/relevances/00005/'


    relevances =[]
    heatmaps = []
    poison_labels =[]
    for root, dirs, files in os.walk(path):

        for name in files:

            #print(name)
            # Save poison labels

            if name.endswith("_poison.npy"):
                poison_labels.append(int(1))
            else:
                poison_labels.append(int(0))


            # Load .npy file
            with open(str(root) + "/" + str(name), 'rb') as f:

                d = np.load(f)

            # Sum over color channels
            d = np.sum(d, axis=0)

            # Save 2D heatmap for input to OT library
            dd = d
            # Transform 32x32 Heatmap into 1024x1 Heatmap
            d = d.reshape((d.shape[0] * d.shape[1]))

            # Save all Heatmaps in one array
            heatmaps.append(dd)
            relevances.append(d)

    # Convert list of relevances to numpy array
    rel_array = np.asarray(relevances)

    #Plot heatmap
    #for i in range(20):
    #    plt.imshow(heatmaps[i])
    #    plt.show()

    print('relarray.shape: ', rel_array.shape)

    #Normalize relevance maps to [0,1] before computing pw. L2 dist:

    rel_array_normalized = []
    for i in range(rel_array.shape[0]):
        r = rel_array[i]
        min = r.min()
        max = r.max()

        # Normalize "by hand"
        r = (r-min)/(max-min)

        # Division by total mass
        r = r/r.sum()

        rel_array_normalized.append(r)

    rel_array_normalized = np.asarray(rel_array_normalized)

    # Zeige normalisierte Heatmap
    #plt.imshow(rel_array_normalized[0].reshape(32,32))
    #plt.show()
    # Remark: Heatmap und normalisierte HEatmap sehen im plot identisch aus
    
    
    #rel_array = np.concatenate(rel_array, axis=2)
    #print(rel_array.shape)
    
    #### Compute distance matrices
    
    # L2-Distance
    
    
    l2_dist = distance_matrix(rel_array_normalized, rel_array_normalized) # Distanzmatrix der pw. Distanzen zwischen den Heatmaps
    print('max: ', l2_dist.max())
    print('min: ', l2_dist.min())
    
    ##### Berechne Distanzen mithilfe von Gromov-Wasserstein
    
    #Wähle erste und zweite Heatmap aus der Liste aus:
    heatmap_array = np.asarray(heatmaps)
    print(heatmap_array.shape)
    im1 = heatmap_array[0]
    im2 = heatmap_array[19]
    
    print('l2:', l2_dist[0][19])
    # Berechne GW-Distanz zwischen beiden Heatmaps
    #n_samples = 32*32
    
    # Speichere Pixelkoordinaten als (x,y)
    # Speichere zusaätzlich Relevanz r passen zum Koordinatenpunkt





    xy1, x1, y1, r1 = heatmap_to_rel_coord(im1)
    xy2, x2, y2, r2 = heatmap_to_rel_coord(im2)
    
    # Compute distance kernels, normalize them and then display
    
    C1 = sp.spatial.distance.cdist(xy1, xy1)
    C2 = sp.spatial.distance.cdist(xy2, xy2)
    
    C1 /= C1.max()
    C2 /= C2.max()
    
    # Compute Gromov-Wasserstein plans and distance
    # https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein
    
    
    #print(gw0,log0)
    # Compute distance matrices for GWD and EGWD:
    
    
    # GWD-Matrix:
    
    
    
    #GWD = compute_GWD_matrix(heatmap_array, method='GWD')
    #EGWD = compute_GWD_matrix(heatmap_array, method='EGWD')
    
    fname_to_save = '/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/LRP_Outputs/incv3_20_epochs_normalizede-rule/EGWD' + ".npy"
    #with open(fname_to_save, 'wb') as f:
    #
    #    np.save(f, EGWD)
    
    # ----------------------------------------------------------------------------------------------------------------------
    
    
    # Compute Affinity matrices/Transformation of distance matrices: -------------------------------------------------------
    #### Compute pw (binary) affinity scores using kNN
    k = 10
    #Ohne kNN: Sortiere jeden Zeile und merke die k Indices(ungleich der Diagonalen) mit der niedrigsten Distanz.
    ind = np.argsort(l2_dist, axis=1)
    #ind = np.argsort(GWD, axis=1)
    #ind = np.argsort(EGWD, axis=1)
    
    #Reduziere die Inidces zeilenweise auf die ersten 10+1 Indices, da der identische Punkt immer mit Abstand 0 dazugehört
    ind = ind[:, 0:k]
    
    #print(ind)
    #print(ind.shape)
    #print(ind.shape[0])
    A = np.zeros_like(l2_dist)
    
    # Setze 1's an die passenden Stellen in A
    for i in range(0, ind.shape[0]): # Zeilen
        for j in range(0, ind.shape[1]): # Spalten
            # Man sollte sich lieber die Indices in ind anschauen und dann
            # an der richtigen Stelle in A einen Eintrag setzen
            #print(i, j)
            A[i, ind[i, j]] = 1
    
    
    # Die Summe über jede Zeile von A ist nun 10, d.h. die 10 nächst gelegenen Punkte sind pro Zeile mit einer 1 vermerkt
    #print(np.sum(A, 1))
    
    # Erzeuge eine symmetrische Affinitätsmatrix
    A = 0.5 * (A + np.transpose(A))
    
    # ----------------------------------------------------------------------------------------------------------------------
    
    # Compute spectral embedding: ------------------------------------------------------------------------------------------
     # compute the symmetrized and normalized p.s.d. graph laplacian
     # COmpute D:
    D = np.zeros_like(A)
    Dinv = np.zeros_like(A)
    #A = np.array([[1,2],[3,4]])
    rowwise_sum = A.sum(1)
    
    
    for i in range(D.shape[0]):
        D[i][i] = A.sum(1)[i]
        Dinv[i][i] = np.sqrt(A.sum(1)[i])
    
    L = D-A
    
    Lsym = Dinv*L*Dinv
    
    
    # Eigenvalue Decomposition of Lsym:
    #TODO: Wahl von k und q?
    #TODO: eigsh vs eigs? #eigsh gibt nur relle Eigenwerte zurück
    l = 30
    
    vals, vecs = eigsh(Lsym, k=l)
    print(vals)
    x = range(1, l+1)
    y = np.sort(np.real(vals))
    
    plt.plot(x, y, 'o')
    plt.title(r'Absolute values of first ' + str(l) + ' eigenvalues of $L_{sym}$ with ' + str(k) + '-nearest neighbours')
    #plt.xlim([0, 25])
    #plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(rel_array_normalized)
    print(kmeans.sum())
    # kmeans wirft 2036 Bilder in die eine Klasse, bei 2463 Bildern entspricht das 82.66 Prozent bzw. 27.33 Prozent
    print(kmeans)

    # Tausche labels, sodass die kleinere Klasse das poison_label=1 besitzt
    clustering_result = 1-np.asarray(kmeans)
    print(clustering_result.sum())
    
    
    print(set(clustering_result))
    print(set(poison_labels))
    # Auswertung des Clusterings für zwei Klassen
    from sklearn.metrics import confusion_matrix
    
    a,b,c,d = confusion_matrix(poison_labels, clustering_result).ravel()
    #specificity = tn / (tn+fp)
    #print(tn, fp, fn, tp)
    # 1650 0 386 427
    print(a, b, c, d)

    #kmeans++ mit euklidischer Distanz



    rel_array_normalized = rel_array_normalized
    #print(poison_labels[0:nn])

    # Suppose we only have to clusters labeled as 0 and 1
    # Choose first index randomly
    n = rel_array_normalized.shape[0] #number of vectors to cluster
    cluster = np.zeros(n)
    seq = list(range(0, n))
    idx1 = random.sample(seq, k=1)[0]
    print('idx1: ', idx1)

    # Compute distances of every other sample to the firt chosen center
    distances_to_first_center = []
    for i in range(n):
        d = np.linalg.norm(rel_array_normalized[idx1]-rel_array_normalized[i])
        distances_to_first_center.append(d)

    distances_to_first_center = np.asarray(distances_to_first_center)
    print(distances_to_first_center.shape)
    print('done.')

    # Set probabilities of choosing next cluster center:
    p = np.square(distances_to_first_center)
    p /= p.sum()
    p[idx1] = 0
    p /= p.sum()

    idx2 = choice(seq, size=1, p=p)[0]
    print('idx_2: ', idx2)

    clustering = np.empty(shape=(n,))
    clustering[:] = np.nan

    clustering[idx1] = 0
    clustering[idx2] = 1

    iter = 0
    max_iter = 5
    print(' ==> Starting k-means++-iterations')
    break_while = False

    #Initialize cluster centers:
    cc = {0: rel_array_normalized[idx1],
          1: rel_array_normalized[idx2]}

    while iter < max_iter:

        print('iter: ', iter)
        print('==> Recalculate distances to each barycenter')

        distances_to_cluster_centers = np.empty(shape=(n, 2))
        distances_to_cluster_centers[:][:] = np.nan

        for i in range(0, n):
            #print('i: ', i)

            distances_to_cluster_centers[i][0] = np.linalg.norm(cc[0] - rel_array_normalized[i])
            distances_to_cluster_centers[i][1] = np.linalg.norm(cc[1] - rel_array_normalized[i])

            #print('==> Update Clustering')
            #print(distances_to_cluster_centers[i])
            #print(distances_to_cluster_centers[i].argmin())
            if distances_to_cluster_centers[i].argmin() == 0:
                clustering[i] = 0
            if distances_to_cluster_centers[i].argmin() == 1:
                clustering[i] = 1

        if iter > 0:
            #Check if clustering has changed in the last iteration:
            if (clustering_old-clustering).sum() == 0:
                #Clustering didnt get updated -> Stop k-means iteration:
                break_while = True

        #print(clustering.sum())
        clustering_old = clustering
        print('Clustering: ', clustering)

        if break_while:
            # break while loop
            print('k-means Clustering didnt change and is being stopped.')
            print('after iteration', iter)
            break

        # Compute new barycenters
        for j in [0, 1]:
            idx = np.where(clustering == j)
            new_cc = rel_array_normalized[idx[0]].mean(axis=0)
            cc[j] = new_cc

        iter += 1

    # Cluster Auswertung:
    if clustering.sum() > clustering.shape[0]/2:
        clustering = 1 - clustering

    print('final_CLustering: ', clustering)
    print(clustering.sum())
    # Evaluate clustering:
    a, b, c, d = confusion_matrix(poison_labels, clustering).ravel()
    #specificity = tn / (tn+fp)
    #print(tn, fp, fn, tp)
    # 1650 0 386 427
    print('(tn, fp, fn, tp): ', a, b, c, d)
