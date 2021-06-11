"""
kmeans++ mit GWD und GW-Barycenters

"""
import concurrent
import os
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
#import ot
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import distance_matrix
from numpy.random import choice
from scipy.sparse.linalg import eigs, eigsh
import sys

from sklearn.metrics import confusion_matrix

#a,b,c,d = confusion_matrix(poison_labels, clustering_result).ravel()
#specificity = tn / (tn+fp)
#print(tn, fp, fn, tp)

#import ot
import time
import torch
import random
import numpy as np
import cupy
#print(torch.rand(10).cuda())

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__version__)
import scipy; print("SciPy", scipy.__version__)
import ot.gpu; print("POT", ot.__version__)
print("Is Cuda available: {}".format(torch.cuda.is_available()))


def rel_coords_to_distance_matrix(xy, x, y, r, threshold):
    r_inds = r.argsort()
    r_sorted = r[r_inds[::-1]]
    xy_sorted = xy[r_inds[::-1]]


    counter = 0
    sum = 0

    #threshold = 0.99
    while sum <= threshold:
        sum += r_sorted[counter]
        counter += 1

    # Compute distance kernels, normalize them and then display

    xy = xy_sorted[:counter].astype(np.float64)
    r = r_sorted[:counter].astype(np.float64)
    p = r/r.sum()

    C = sp.spatial.distance.cdist(xy, xy).astype(np.float64)
    C /= C.max()

    return C, p


def compute_barycenter_from_images(images, threshold=0.99, n_samples=50):

    """
    Computes the 'average' of n input images in the sense of Wasserstein distance.


    Parameters
    ----------
    images : ndarray, shape (ns, ns)
        input heatmaps of size (ns, ns)
    threshold : float
          percentage up to which pixels from original image with total weight <= threshold are taken
    n_samples :  int


    Returns
    -------
    embedding :
           Embedded barycenter
    """

    Cs = []  # cost matrices
    ps = []
    #n_samples = 50
    p = ot.unif(n_samples)
    lambdas = []

    for im in images:

        xy, x, y, r = heatmap_to_rel_coord(im)

        r_inds = r.argsort()
        r_sorted = r[r_inds[::-1]]
        xy_sorted = xy[r_inds[::-1]]

        counter = 0
        sum = 0

        while sum <= threshold:
            sum += r_sorted[counter]
            counter += 1

        # Compute distance kernels, normalize them and then display

        xy = xy_sorted[:counter].astype(np.float64)
        r = r_sorted[:counter].astype(np.float64)
        r /= r.sum()

        C = sp.spatial.distance.cdist(xy, xy).astype(np.float64)
        C /= C.max()
        Cs.append(C)

        ps.append(r)
        lambdas.append(1.0)

    #lambdas = np.asarray(lambdas)
    #lambdas /= np.float(len(lambdas))

    lambdas = [float(i)/len(lambdas) for i in lambdas]

    # Berechne Barycenter mit POT-Funktion
    barycenter = ot.gromov.gromov_barycenters(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss')

    return barycenter


def compute_barycenter_from_measured_distances(measured_distances, id, n_samples=50):

    """
    Computes the 'average' of n measured distance matrices in the sense of Wasserstein distance.


    Parameters
    ----------
    measured_distances :
        list of tuples (C,p)

    n_samples :  int
        number of points in the computed barycenter

    Returns
    -------
    embedding :
           (Embedded) barycenter
    """
    """
    Cs = []  # cost matrices
    ps = []

    p = ot.unif(n_samples)
    lambdas = []

    for (C, p) in measured_distances:

        Cs.append(C)

        ps.append(p)
        lambdas.append(1.0)
    """

    md = measured_distances
    idx = np.where(clustering == id)
    #print('idx: ', idx[0][0])
    #print(md[:, 0][idx[0]].shape)

    Cs = md[:, 0][idx[0]]
    print(Cs.shape)
    p = ot.unif(n_samples)
    ps = md[:, 1][idx[0]]
    print(ps.shape)
    #lambdas = np.asarray(lambdas)
    #lambdas /= np.float(len(lambdas))

    lambdas = np.ones(Cs.shape[0])
    lambdas /= lambdas.sum()
    print(lambdas.shape)

    # Berechne Barycenter mit POT-Funktion
    barycenter = ot.gromov.gromov_barycenters(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss')

    return barycenter


def heatmap_to_rel_coord(im):
        """Transform Relevance map to list of coordinates and relevances per pixel in the range of [0,1]

        Input:
        -------
                    im: 2D image with one channel

        Returns:
        -----------
        xy = list of tuples (x,y)
        x = list of x-coordinates
        y = list of y-coordinates
        r = list of mass per point
        """

        x = []
        y = []
        r = []
        xy = []
        #for i in range(31, 0, -1):
        for i in range(32):
            for j in range(32):
                x.append(j)
                y.append(i)
                r.append(im[j][i])
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


def smacof_mds(C, dim, max_iter=3000, eps=1e-9):
        """
        Returns an interpolated point cloud following the dissimilarity matrix C
        using SMACOF multidimensional scaling (MDS) in specific dimensioned
        target space

        Parameters
        ----------
        C : ndarray, shape (ns, ns)
            dissimilarity matrix
        dim : int
              dimension of the targeted space
        max_iter :  int
            Maximum number of iterations of the SMACOF algorithm for a single run
        eps : float
            relative tolerance w.r.t stress to declare converge

        Returns
        -------
        npos : ndarray, shape (R, dim)
               Embedded coordinates of the interpolated point cloud (defined with
               one isometry)
        """

        rng = np.random.RandomState(seed=3)

        mds = manifold.MDS(
            dim,
            max_iter=max_iter,
            eps=1e-9,
            dissimilarity='precomputed',
            n_init=1)
        pos = mds.fit(C).embedding_

        nmds = manifold.MDS(
            2,
            max_iter=max_iter,
            eps=1e-9,
            dissimilarity="precomputed",
            random_state=rng,
            n_init=1)
        npos = nmds.fit_transform(C, init=pos)

        return npos


def heatmap_to_distance_matrix(im, threshold=0.99):
        xy, x, y, r = heatmap_to_rel_coord(im)
        C, p = rel_coords_to_distance_matrix(xy, x, y, r, threshold=threshold)

        return C, p


if __name__ == '__main__':

    number_samples = 90 #number of samples per class to check for poisoning attack

    """
    clustering = np.array([1, 1, 0, 0])

    print('Clustering: ', clustering)
    with open('/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/md.npy', 'rb') as f:

        md = np.load(f, allow_pickle=True)

    print(md[1][0].shape)
    """
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
    rel_array = np.asarray(relevances).astype(np.float64)
    rel_array = rel_array[0:number_samples]
    heatmaps = heatmaps[0:number_samples]
    poison_labels = poison_labels[0:number_samples]

    print(poison_labels)
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


    #Wähle 2 images, für die ein Barycenter berechnet werden soll:
    #Wähle erste und zweite Heatmap aus der Liste aus:
    heatmap_array = np.asarray(heatmaps).astype(np.float64)
    print('HEATMAPSARRAYSHAOPE:', heatmap_array.shape[0])

    im1 = heatmap_array[1]  #77
    im2 = heatmap_array[2]  #88 #beides Heatmaps mit Sticker

    # Speichere Bilder als png
    #matplotlib.image.imsave('poison3.png', im1)
    #matplotlib.image.imsave('poison4.png', im2)

    #plt.imshow(im1)
    #plt.show()
    #plt.imshow(im2)
    #plt.show()

    print('Min: ', im1.min())
    print('Max: ', im1.max())
    #print(im1)
    print(im2.min())

    xy1, x1, y1, r1 = heatmap_to_rel_coord(im1)
    xy2, x2, y2, r2 = heatmap_to_rel_coord(im2)


    print('argmax: ', np.argmax(r1))
    print('MAx, x, y: ', x1[np.argmax(r1)], y1[np.argmax(r1)])
    #r2_sorted = np.sort(r2)[::-1]
    #print(r2_sorted)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Heatmap und erzeugte Punktwolke')
    axs[0, 0].imshow(im1)
    axs[1, 0].imshow(im2)


    r2_inds = r2.argsort()
    r2_sorted = r2[r2_inds[::-1]]
    xy2_sorted = xy2[r2_inds[::-1]]

    r1_inds = r1.argsort()
    r1_sorted = r1[r1_inds[::-1]]
    xy1_sorted = xy1[r1_inds[::-1]]

    counter2 = 0
    sum = 0

    threshold = 0.99
    while sum <= threshold:
        sum += r2_sorted[counter2]
        counter2 += 1

    sum1 = 0
    counter1 = 0

    while sum1 < threshold:
        sum1 += r1_sorted[counter1]
        counter1 += 1

    print('counter2: ', counter2)
    print('counter1: ', counter1)


    #sys.exit()
    # Compute distance kernels, normalize them and then display

    xy1 = xy1_sorted[:counter1].astype(np.float64)
    xy2 = xy2_sorted[:counter2].astype(np.float64)
    r1 = r1_sorted[:counter1].astype(np.float64)
    r2 = r2_sorted[:counter2].astype(np.float64)



    # Plot point cloud of first image
    axs[0, 1].scatter(xy1[:, 0], xy1[:, 1])
    axs[0, 1].set_xlim((0, 32))
    axs[0, 1].set_ylim((32, 0))
    axs[0, 1].set_aspect('equal')

    axs[1, 1].scatter(xy2[:, 0], xy2[:, 1])
    axs[1, 1].set_xlim((0, 32))
    axs[1, 1].set_ylim((32, 0))
    #plt.setp(axs, ylim=axs[0, 0].get_ylim())
    axs[1, 1].set_aspect('equal')
    #plt.show()
    #TODO: Der plot sieht jetzt zwar passend aus, werden die Relevanzen in rel_to_heatmap in der richtigen Reihenfolge ausgelesen?


    print('MINMIN: ', r1.min())

    C1 = sp.spatial.distance.cdist(xy1, xy1).astype(np.float64)
    C2 = sp.spatial.distance.cdist(xy2, xy2).astype(np.float64)

    C1 /= C1.max()
    C2 /= C2.max()

    #C1 /= C1.sum()
    #C2 /= C2.sum()

    print(C2.shape)

    # Setze Gewichte gleichverteilt
    #TODO: Nehme ich r1,r2 (oben), d.h. verschieden gewichtet, erhalte ich eine Fehlermedlung bei der Berechnung von bary
    #r1 = ot.unif(len(xy1))
    #r2 = ot.unif(len(xy2))

    #Idee: Normiere r1 und r2 so, dass die Summe 1 ergibt
    r1 /= r1.sum()
    r2 /= r2.sum()
    print('SUMSUM: ', r2.sum())
    lambdas = [0.5, 0.5]
    # Idee funktioniert

    n_samples = 10
    p = ot.unif(n_samples)
    #TODO: n_samples gibt auf der einen Seit an, wie viele Punkte ausgewählt werden sollen, andererseits ist das aber auch die Dimension (n_samples,n_samples) des bary_centers
    # Wie passt das zusammen?
    #Compute barycenter
    bary = ot.gromov.gromov_barycenters(n_samples, [C1, C2], [r1, r2], p, lambdas, 'square_loss', max_iter=100, tol=1e-3, verbose=False)
    print(bary.max())
    print(bary.min())

    print(bary.shape)
    #print(bary)
    plt.imshow(bary)
    #plt.show()

    clf = PCA(n_components=2)

    embedding = clf.fit_transform(smacof_mds(bary, 2))
    #embedding = smacof_mds(bary, 32)

    plt.scatter(embedding[:, 0], embedding[:, 1], color='r')
    plt.title('Embedding')
    #plt.show()
    #print(embedding)



    # Sampled original image


    npos1 = smacof_mds(C1, 2)
    npos2 = smacof_mds(C2, 2)

    plt.scatter(npos1[:, 0], npos1[:, 1])
    plt.title('Embedded distance matrix of first original image')
    #plt.show()




    """
    bary = compute_barycenter_from_images(heatmap_array[1:2])
    
    
    #clf = PCA(n_components=2)
    
    embedding = clf.fit_transform(smacof_mds(bary, 2))
    plt.scatter(embedding[:, 0], embedding[:, 1], color='r')
    plt.title('Embedding Durschnitt')
    plt.show()
    """

    #TODO: Vergleich GWD zwischen barycenter und embedding eines Ursprungsbildes

    #### kmeans++ ####
    print('#### kmeans++ ####')
    # Initialisierung
    # Lege gewünschte Cluster-Anzahl k fest:
    k = 2
    max_iter = 3  # number of maximum iterations
    #cluster_centers = np.empty(shape=(k, 2))
    #cluster_centers[:][:] = np.nan
    cluster_centers = [[[], []], [[], []]]
    print('cc: ', cluster_centers)
    # Wähle im ersten Schritt eine zufällige Heatmap als erstes Zentrum
    n = heatmap_array.shape[0]  # number of heatmaps to cluster
    cluster = np.zeros(n)
    print('Hallo', n)
    seq = list(range(0, n))
    #print(seq)

    idx_1 = random.sample(seq, k=1)[0]
    print('idx1: ', idx_1)

    # Compute distances of every other sample to the chosen first center
    measured_distances = np.asarray([heatmap_to_distance_matrix(im) for im in heatmap_array])
    print(measured_distances.shape)
    #np.expand_dims(measured_distances, axis=1)
    #print(measured_distances.shape)
    print(measured_distances[idx_1][1].shape)
    distances = []
    #sys.exit()

    print(' ==> Compute distances to first mean')
    for i in range(0, n):
        print('i: ', i)

        gw, log = ot.gromov.entropic_gromov_wasserstein2(
                measured_distances[idx_1][0], measured_distances[i][0],
                measured_distances[idx_1][1], measured_distances[i][1], 'square_loss', epsilon=5e-4, log=True)
        print(log['gw_dist'])
        distances.append(log['gw_dist'])

        #distances.append(1.0)

    distances = np.asarray(distances)
    distances /= distances.sum()
    print(distances.shape[0])
    """
    with open('distances.npy', 'wb') as f:

            np.save(f, distances)

    with open('/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/distances.npy', 'rb') as f:

            distances = np.load(f, allow_pickle=True)
    """

    ##### Initialization #####
    # Choose probabilities of choosing next k-1 centers:
    p = np.square(distances)
    p /= p.sum()
    print(p)
    #seq2 = np.delete(seq, np.where(seq == idx))
    #TODO: Setze Wahrscheinlichkeit an Stelle des ersten indexes auf 0 und normalisiere anschließend
    p[idx_1] = 0
    p /= p.sum()
    print(p)
    idx_2 = choice(seq, size=1, p=p)[0]
    print('idx_2: ', idx_2)

    clustering = np.empty(shape=(n,))
    clustering[:] = np.nan

    clustering[idx_1] = 0
    clustering[idx_2] = 1

    #Use dict for cluster centers cc:
    cc = {0: {'dist_m': measured_distances[idx_1][0], 'weights': measured_distances[idx_1][1]},
          1: {'dist_m': measured_distances[idx_2][0], 'weights': measured_distances[idx_2][1]}}

    iter = 0
    print(' ==> Starting k-means++-iterations')
    while iter < max_iter:
        iter += 1
        print('iter: ', iter)
        print('==> Recalculate distances to each barycenter')
        # Compute for every point, which is not part of a cluster the distance to the cluster centers 1 and 2, take the minimum and assign the point accordingly.
        distances_to_cluster_centers = np.empty(shape=(n, 2))
        distances_to_cluster_centers[:][:] = np.nan
        """
        if iter == 1:
            clustering = np.array([1, 1, 0, 0])
            with open('/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/md.npy', 'rb') as f:

                md = np.load(f, allow_pickle=True)

            measured_distances = md
        else:
        """

        # Compute distances to all barycenters per data point

        start_time = time.time()

        for i in range(0, n):
            print('i: ', i)
            #print(cc[0]['weights'].shape)
            #print(cc[1]['weights'].shape)#

            #print(cc[0]['dist_m'].shape)
            #print(cc[1]['dist_m'].shape)

            gw, log = ot.gromov.entropic_gromov_wasserstein2(
                    cc[0]['dist_m'], measured_distances[i][0],
                    cc[0]['weights'], measured_distances[i][1], 'square_loss', epsilon=5e-4, log=True)
            distances_to_cluster_centers[i][0] = log['gw_dist']

            print('done.')
            gw, log = ot.gromov.entropic_gromov_wasserstein2(
                    cc[1]['dist_m'], measured_distances[i][0],
                    cc[1]['weights'], measured_distances[i][1], 'square_loss', epsilon=5e-4, log=True)
            distances_to_cluster_centers[i][1] = log['gw_dist']
            print('done.')

            print('Distances to centers: ', distances_to_cluster_centers)
            #Choose minimum distance and set cluster label accordingly:
            print('==> Update Clustering')
            print(distances_to_cluster_centers[i])
            print(distances_to_cluster_centers[i].argmin())
            if distances_to_cluster_centers[i].argmin() == 0:
                clustering[i] = 0
            if distances_to_cluster_centers[i].argmin() == 1:
                clustering[i] = 1

            print('Clustering: ', clustering)

            """
            with open('clustering.npy', 'wb') as f:

                np.save(f, clustering)

            with open('md.npy', 'wb') as f:

                np.save(f, measured_distances)
            """
        print("--- %s seconds ---" % (time.time() - start_time))

        # Recalculate barycenters
        start_time = time.time()
        print('==> Recalculate barycenters per cluster')

        # Compute barycenter per cluster and update cluster center:
        bary1 = compute_barycenter_from_measured_distances(measured_distances=measured_distances, id=0)

        clf = PCA(n_components=2)
        embedding = clf.fit_transform(smacof_mds(bary1, 2))

        # Create measured distance matrix of barycenter
        C = sp.spatial.distance.cdist(embedding, embedding).astype(np.float64)
        #C /= C.max()
        C /= C.sum()
        p = ot.unif(C.shape[0])

        cc[0]['dist_m'] = C
        cc[0]['weights'] = p

        bary2 = compute_barycenter_from_measured_distances(measured_distances=measured_distances, id=1)

        clf = PCA(n_components=2)
        embedding = clf.fit_transform(smacof_mds(bary2, 2))

        # Create measured distance matrix of barycenter
        C = sp.spatial.distance.cdist(embedding, embedding).astype(np.float64)
        #C /= C.max()
        C /= C.sum()
        p = ot.unif(C.shape[0])

        cc[1]['dist_m'] = C
        cc[1]['weights'] = p

        print("--- %s seconds ---" % (time.time() - start_time))

    print('final_CLustering: ', clustering)
    # Evaluate clustering:
    a, b, c, d = confusion_matrix(poison_labels, clustering).ravel()
    #specificity = tn / (tn+fp)
    #print(tn, fp, fn, tp)
    # 1650 0 386 427
    print('(tn, fp, fn, tp): ', a, b, c, d)

    sys.exit()
# Compute Gromov-Wasserstein plans and distance
# https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein


#print(gw0,log0)
# Compute distance matrices for GWD and EGWD:

sys.exit()
# GWD-Matrix:
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
                gw, log = ot.gromov.entropic_gromov_wasserstein2(
                C1, C2, r1, r2, 'square_loss', epsilon=5e-4, log=True, verbose=verbose)
            else:
                print('METHOD not implemented.')

            GWD[i][j] = log['gw_dist']

            #Use symmetry
            GWD[j][i] = GWD[i][j]


        # Setze Diagonalelement =0
        GWD[i][i] = 0

    return GWD


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
from sklearn.cluster import KMeans


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
