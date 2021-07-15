import numpy as np
import ot
import scipy as sp
from sklearn import manifold


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


def compute_barycenter_from_Cp(CC, pp, clustering, id, n_samples=400):


    idd = np.where(clustering==id)
    Cs = CC[idd]
    ps = pp[idd]
    p = ot.unif(n_samples)

    lambdas = np.ones_like(ps)
    lambdas = [float(i)/len(lambdas) for i in lambdas]

    # Berechne Barycenter mit POT-Funktion
    barycenter = ot.gromov.gromov_barycenters(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss')

    return barycenter

def compute_barycenter_from_images(images, threshold=0.99, n_samples=400):

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

        # Compute distance kernels, normalize them

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


def compute_barycenter_from_measured_distances(measured_distances, clustering, id, n_samples=50):

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
    print('mdshape: ', md.shape)
    print(md[:, 0].shape)
    print(idx[0].shape)
    print(md[:, 0][idx[0]].shape)

    print(idx[0][0:1])
    Cs = md[:, 0][idx[0]]

    print(Cs.shape)

    p = ot.unif(n_samples)
    ps = md[:, 1][idx[0]]
    print(ps.shape)
    #lambdas = np.asarray(lambdas)
    #lambdas /= np.float(len(lambdas))

    lambdas = np.ones(Cs.shape[0])
    lambdas /= lambdas.sum()
    #print(lambdas.shape)

    lambdas = list(lambdas)

    # Berechne Barycenter mit POT-Funktion
    barycenter = ot.gromov.gromov_barycenters(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss')
    print('Bary computed.')

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


def compute_GWD_to_index(cp1, cp2):

        C1 = cp1[0]
        p1 = cp1[1]
        C2 = cp2[0]
        p2 = cp2[1]

        gw, log = ot.gromov.entropic_gromov_wasserstein2(
                C1, C2,
                p1, p2, 'square_loss', epsilon=5e-4, log=True)

        return log['gw_dist']

