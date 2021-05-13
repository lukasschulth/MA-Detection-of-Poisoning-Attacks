

# Open images of suspicious class
path = '/home/lukasschulth/Documents/MA-Detection-of-Poisoning-Attacks/coding/LRP_Outputs/incv3_matthias_v2e-rule/relevances/00026/'


import os

import matplotlib.pyplot as plt

from PIL import Image
import ot
import scipy as sp
import numpy as np
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import distance_matrix

relevances =[]
heatmaps = []
for root, dirs, files in os.walk(path):

    for name in files:


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
# plt.imshow(heatmaps[0])

print(rel_array.shape)
#rel_array = np.concatenate(rel_array, axis=2)
#print(rel_array.shape)

#### Compute distance matrices

# L2-Distance


l2_dist = distance_matrix(rel_array, rel_array)


# TODO: Gromov-Wasserstein-Distance


#### Compute pair_wise (binary) affinity scores using kNN
k = 10
#Ohne kNN: Sortiere jeden Zeile und merke die k Indices(ungleich der Diagonalen) mit der niedrigsten Distanz.
ind = np.argsort(l2_dist, axis=1)

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

##### Berechne Distanzen mithilfe von Gromov-Wasserstein

#Wähle erste und zweite Heatmap aus der Liste aus:
heatmap_array = np.asarray(heatmaps)
print(heatmap_array.shape)
im1 = heatmap_array[0]
im2 = heatmap_array[17]

# Berechne GW-Distanz zwischen beiden Heatmaps
#n_samples = 32*32
#TODO: Normalisiere jedes Bild

# Speichere Pixelkoordinaten als (x,y)
# Speichere zusaätzlich Relevanz r passen zum Koordinatenpunkt


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
            r.append(im1[i][j])
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


xy1, x1, y1, r1 = heatmap_to_rel_coord(im1)
xy2, x2, y2, r2 = heatmap_to_rel_coord(im2)


n_samples = 30  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4, 4])
cov_t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)
#print(xs)
P = sp.linalg.sqrtm(cov_t)
xt = np.random.randn(n_samples, 3).dot(P) + mu_t
#print(xt)

fig = pl.figure()
ax1 = fig.add_subplot(121)
ax1.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(xt[:, 0], xt[:, 1], xt[:, 2], color='r')
pl.show()

# Compute distance kernels, normalize them and then display

C1 = sp.spatial.distance.cdist(xy1, xy1)
C2 = sp.spatial.distance.cdist(xy2, xy2)

C1 /= C1.max()
C2 /= C2.max()

pl.figure()
pl.subplot(121)
pl.imshow(C1)
pl.subplot(122)
pl.imshow(C2)
pl.show()

# Compute Gromov-Wasserstein plans and distance
# https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.gromov_wasserstein
p = r1
#print(p.sum())
q = r2
#print(q.sum())
gw0, log0 = ot.gromov.gromov_wasserstein(
    C1, C2, p, q, 'square_loss', verbose=True, log=True)

gw, log = ot.gromov.entropic_gromov_wasserstein(
    C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)


print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
print('Entropic Gromov-Wasserstein distances: ' + str(log['gw_dist']))


pl.figure(1, (10, 5))

pl.subplot(1, 2, 1)
pl.imshow(gw0, cmap='jet')
pl.title('Gromov Wasserstein')

pl.subplot(1, 2, 2)
pl.imshow(gw, cmap='jet')
pl.title('Entropic Gromov Wasserstein')

#pl.show()




# Compute spectral embedding:



