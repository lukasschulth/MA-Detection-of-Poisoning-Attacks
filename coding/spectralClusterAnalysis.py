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

#TODO: Im Fall von zwei Clustern ist die Genauigkeit bereits nach der Initialisierung sehr gut
# Wann stoppt kmeans? Wie mache ich das für k>2 Cluster?
# Kann ich die Berechnung der barycenter pro Cluster parallelisieren?
