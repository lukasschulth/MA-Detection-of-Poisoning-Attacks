import numpy as np
import ot
import scipy as sp
import torch
from sklearn import manifold
#sdsdsdsd
from scipy.spatial.distance import cdist
import ot
from bregman_torch import sinkhorn_torch
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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



def compute_barycenter_from_Cp(CC, pp, clustering, id, n_samples=400, entropic=False, eps=0.1):


    idd = np.where(clustering==id)
    Cs = CC[idd]
    ps = pp[idd]
    p = ot.unif(n_samples)

    lambdas = np.ones_like(ps)
    lambdas = [float(i)/len(lambdas) for i in lambdas]

    # Berechne Barycenter mit POT-Funktion
    if entropic:
        barycenter = entropic_gromov_barycenters(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss', eps=eps)
    else:
        barycenter = ot.gromov.gromov_barycenters(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss')

    return barycenter

def compute_barycenter_from_Cp_torch(CC, pp, id, n_samples=400, entropic=False, eps=0.1):

    Cs = CC[id[0][0]]
    ps = pp[id[0][0]]

    p = ot.unif(n_samples)
    p = torch.from_numpy(p)

    lambdas = torch.ones_like(ps)
    lambdas = [float(i)/len(lambdas) for i in lambdas]

    # Berechne Barycenter mit POT-Funktion
    if entropic:
        barycenter = entropic_gromov_barycenters_torch(n_samples, Cs, ps, p, lambdas=lambdas, loss_fun='square_loss', eps=eps)
    else:
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

def heatmap_to_rel_coord_gpu(im):
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
        xy = torch.tensor(xy)
        x = torch.tensor(x)
        y = torch.tensor(y)
        r = torch.tensor(r)

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
        print('test')
        gw, log = ot.gromov.entropic_gromov_wasserstein2(
                C1, C2,
                p1, p2, 'square_loss', epsilon=5e-4, log=True)

        return log['gw_dist']

def entropic_gromov_barycenters(N, Cs, ps, p, lambdas, loss_fun,
                       max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None,eps=0.02):
    """
    Returns the gromov-wasserstein barycenters of S measured similarity matrices

    (Cs)_{s=1}^{s=S}

    The function solves the following optimization problem with block
    coordinate descent:

    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)

    Where :

    - Cs : metric cost matrix
    - ps  : distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S np.ndarray of shape (ns, ns)
        Metric cost matrices
    ps : list of S np.ndarray of shape (ns,)
        Sample weights in the S spaces
    p : ndarray, shape (N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    loss_fun :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | ndarray, shape(N,N)
        Random initial value for the C matrix provided by user.

    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    S = len(Cs)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        # XXX : should use a random state and not use the global seed
        xalea = np.random.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while(err > tol and cpt < max_iter):
        Cprev = C

        T = [ot.gromov.entropic_gromov_wasserstein(Cs[s], C, ps[s], p, loss_fun,
                                epsilon=eps) for s in range(S)]

        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return C

def entropic_gromov_barycenters_torch(N, Cs, ps, p, lambdas, loss_fun,
                       max_iter=1000, tol=1e-9, verbose=False, log=False, init_C=None,eps=0.02):
    """
    Returns the gromov-wasserstein barycenters of S measured similarity matrices

    (Cs)_{s=1}^{s=S}

    The function solves the following optimization problem with block
    coordinate descent:

    .. math::
        C = argmin_C\in R^NxN \sum_s \lambda_s GW(C,Cs,p,ps)

    Where :

    - Cs : metric cost matrix
    - ps  : distribution

    Parameters
    ----------
    N : int
        Size of the targeted barycenter
    Cs : list of S np.ndarray of shape (ns, ns)
        Metric cost matrices
    ps : list of S np.ndarray of shape (ns,)
        Sample weights in the S spaces
    p : ndarray, shape (N,)
        Weights in the targeted barycenter
    lambdas : list of float
        List of the S spaces' weights
    loss_fun :  tensor-matrix multiplication function based on specific loss function
    update : function(p,lambdas,T,Cs) that updates C according to a specific Kernel
             with the S Ts couplings calculated at each iteration
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : bool | ndarray, shape(N,N)
        Random initial value for the C matrix provided by user.

    Returns
    -------
    C : ndarray, shape (N, N)
        Similarity matrix in the barycenter space (permutated arbitrarily)

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    S = len(Cs)




    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        # XXX : should use a random state and not use the global seed
        xalea = np.random.randn(N, 2)
        C = dist(xalea, xalea)
        C /= C.max()
    else:
        C = init_C

    cpt = 0
    err = 1

    error = []

    while(err > tol and cpt < max_iter):
        Cprev = C

        T = [entropic_gromov_wasserstein_torch(Cs[s], C, ps[s], p, loss_fun,
                                epsilon=eps) for s in range(S)]

        if loss_fun == 'square_loss':
            C = update_square_loss(p, lambdas, T, Cs)

        elif loss_fun == 'kl_loss':
            C = update_kl_loss(p, lambdas, T, Cs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    return C


def update_square_loss(p, lambdas, T, Cs):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration

    Parameters
    ----------
    p : ndarray, shape (N,)
        Masses in the targeted barycenter.
    lambdas : list of float
        List of the S spaces' weights.
    T : list of S np.ndarray of shape (ns,N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape(ns,ns)
        Metric cost matrices.

    Returns
    ----------
    C : ndarray, shape (nt, nt)
        Updated C matrix.
    """

    """
    tmpsum = 0
    print(len(T))
    print(len(lambdas))
    print(len(Cs))

    
    for s in range(len(T)):
        print('s: ', s)
        print(T[s].shape)
        print(Cs[s].shape)
        tmpsum += lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_kl_loss(p, lambdas, T, Cs):
    """
    Updates C according to the KL Loss kernel with the S Ts couplings calculated at each iteration


    Parameters
    ----------
    p  : ndarray, shape (N,)
        Weights in the targeted barycenter.
    lambdas : list of the S spaces' weights
    T : list of S np.ndarray of shape (ns,N)
        The S Ts couplings calculated at each iteration.
    Cs : list of S ndarray, shape(ns,ns)
        Metric cost matrices.

    Returns
    ----------
    C : ndarray, shape (ns,ns)
        updated C matrix
    """
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.exp(np.divide(tmpsum, ppt))


def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist

    Parameters
    ----------

    x1 : ndarray, shape (n1,d)
        matrix with n1 samples of size d
    x2 : array, shape (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str | callable, optional
        Name of the metric to be computed (full list in the doc of scipy),  If a string,
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.


    Returns
    -------

    M : np.array (n1,n2)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    return cdist(x1, x2, metric=metric)


def euclidean_distances(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.
    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)
    """
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]
    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    return distances if squared else np.sqrt(distances, out=distances)

def entropic_gromov_wasserstein2_torch(C1, C2, p, q, loss_fun, epsilon,
                                 max_iter=1000, tol=1e-9, verbose=False, log=False, method='sinkhorn'):
    """
    Returns the entropic gromov-wasserstein discrepancy between the two measured similarity matrices

    (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    - H  : entropy

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
        Distribution in the source space
    q :  ndarray, shape (nt,)
        Distribution in the target space
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    gw, logv = entropic_gromov_wasserstein_torch(
        C1, C2, p, q, loss_fun, epsilon, max_iter, tol, verbose, log=True,method=method)

    logv['T'] = gw

    if log:
        return logv['gw_dist'], logv
    else:
        return logv['gw_dist']

def entropic_gromov_wasserstein_torch(C1, C2, p, q, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False, method='sinkhorn'):
    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)

    (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

        s.t. T 1 = p

             T^T 1= q

             T\geq 0

    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices
    - H  : entropy

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
        Distribution in the source space
    q :  ndarray, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.

    Returns
    -------
    T : ndarray, shape (ns, nt)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    T = torch.outer(p, q)  # Initialization

    constC, hC1, hC2 = init_matrix_torch(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad_torch(constC, hC1, hC2, T)

        T = sinkhorn_torch(p, q, tens, epsilon, method=method)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = torch.linalg.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        log['gw_dist'] = gwloss_torch(constC, hC1, hC2, T)
        return T, log
    else:
        return T


def init_matrix_torch(C1, C2, p, q, loss_fun='square_loss'):
    """Return loss matrices and tensors for Gromov-Wasserstein fast computation

    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in [12]

    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces

    The square-loss function L(a,b)=|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2*b

    The kl-loss function L(a,b)=a*log(a/b)-a+b is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=a*log(a)-a
            * f2(b)=b
            * h1(a)=a
            * h2(b)=log(b)

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
        Coupling between source and target spaces
    p : ndarray, shape (ns,)

    Returns
    -------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)
    constC1 = torch.mm(torch.mm(f1(C1), p.reshape(-1, 1)),
                       torch.ones(len(q), dtype=torch.float64).reshape(1, -1).to(device))

    constC2 = torch.mm(torch.ones(len(p), dtype=torch.float64).reshape(-1, 1).to(device),torch.mm(q.reshape(1, -1), f2(C2).T))
    """
    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    
    """
    constC = constC1 + constC2

    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2

def gwloss_torch(constC, hC1, hC2, T):
    """Return the Loss for Gromov-Wasserstein

    The loss is computed as described in Proposition 1 Eq. (6) in [12].

    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
        Current value of transport matrix T

    Returns
    -------
    loss : float
        Gromov Wasserstein loss

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    tens = tensor_product_torch(constC, hC1, hC2, T)

    return torch.sum(tens * T)


def gwggrad_torch(constC, hC1, hC2, T):
    """Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in [12].

    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
        Current value of transport matrix T

    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    return 2 * tensor_product_torch(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor



def tensor_product_torch(constC, hC1, hC2, T):
    """Return the tensor for Gromov-Wasserstein fast computation

    The tensor is computed as described in Proposition 1 Eq. (6) in [12].

    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
        h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
        h2(C) matrix in Eq. (6)

    Returns
    -------
    tens : ndarray, shape (ns, nt)
        \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    #A = -np.dot(hC1, T).dot(hC2.T)
    A = - torch.mm(torch.mm(hC1, T), hC2.T)
    tens = constC + A
    # tens -= tens.min()
    return tens


