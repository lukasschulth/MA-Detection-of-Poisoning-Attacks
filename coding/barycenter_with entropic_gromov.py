import concurrent
from copy import deepcopy
import multiprocessing
import os
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import distance_matrix
from numpy.random import choice
from scipy.sparse.linalg import eigs, eigsh
import sys
from sklearn.metrics import confusion_matrix
from gwd_utils import *
import time
import torch; print('torch', torch.__version__)
import random

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__version__)
import scipy; print("SciPy", scipy.__version__)
import ot; print("POT", ot.__version__)
print("Is Cuda available: {}".format(torch.cuda.is_available()))


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    random.seed(seed) #seed=0 => idx1=24
    parallel_computation = False
    threshold = 0.99
    num_samples_to_check = 0  #number of samples per class to check for poisoning attack. if ==0 => take all samples
    max_iter = 5  # number of maximum iterations in kmeans algorithm
    n_samples = 10  # number of points in barycenter:
    de = 18 # Paramter um das Auswahlfenster der Länge num_samples_to_check zu verschieben
    eps_init = 0.02 #5e-4'MyFile5.txt'
    eps_update = 0.02

    class_to_check = 5
    verbose = False

    # Ein- und Ausgabe
    #'SPA_incV3_s2_pp015'
    model_names = ['SPA_incV3_s2_pp01']#['SPA_incV3_s2_pp001']#
    model_names = [s + 'e-rule' for s in model_names]

    for model_name in model_names:
        print(model_name)
        file_name = 'clustering_' + model_name + '.txt'

        # Erstelle Ordner für die Ausgaben
        path = os.getcwd()
        path_new = path + "/Clustering_Outputs/"
        if not os.path.exists(path_new):
            os.makedirs(path_new)
        file_name = path_new + file_name

        # Open images of suspicious class
        path = './LRP_Outputs/' + str(model_name) + '/relevances/' + str(class_to_check).zfill(5) + '/'

        relevances = []
        heatmaps = []
        poison_labels = []

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

        if num_samples_to_check > 0:
            rel_array = rel_array[de: de + num_samples_to_check]
            heatmaps = heatmaps[de: de + num_samples_to_check]
            poison_labels = poison_labels[de: de + num_samples_to_check]

        print('poisonLabels: ', poison_labels)
        print('Anzahl an korrumpierten Datenpunkten in der Teilmenge: ', np.asarray(poison_labels).sum())
        #Plot heatmap
        #for i in range(20):
        #    plt.imshow(heatmaps[i])
        #    plt.show()



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

        #### kmeans++ ####
        ##### Initialization ##### -----------------------------------------------------------------------------------------
        print('#### kmeans++ ####')
        # Initialisierung
        # Lege gewünschte Cluster-Anzahl k fest:
        num_cluster_kmeans = 2

        #cluster_centers = np.empty(shape=(k, 2))
        #cluster_centers[:][:] = np.nan
        cluster_centers = [[[], []], [[], []]]
        #print('cc: ', cluster_centers)

        # Wähle im ersten Schritt eine zufällige Heatmap als erstes Zentrum
        n = heatmap_array.shape[0]  # number of heatmaps to cluster
        cluster = np.zeros(n)

        seq = list(range(0, n))
        #print(seq)

        idx_1 = random.sample(seq, k=1)[0]
        #print('idx1: ', idx_1)

        # Compute distances of every other sample to the chosen first center
        #measured_distances = np.asarray([heatmap_to_distance_matrix(im) for im in heatmap_array])

        CC = []
        pp = []
        max_dim = 0
        min_dim = np.square(heatmap_array[0].shape[0])

        for i in range(n):
            c_, p_ = heatmap_to_distance_matrix(heatmap_array[i])
            CC.append(c_)
            pp.append(p_)
            if c_.shape[0] > max_dim:
                max_dim = c_.shape[0]
            if c_.shape[0] < min_dim:
                min_dim = c_.shape[0]

        CC = np.asarray(CC)
        pp = np.asarray(pp)


        #print('max_dim: ', max_dim)
        #print('min_dim:', min_dim)

        def my_function_star(args):
            return compute_GWD_to_index(*args)


        print(' ==> Compute distances to first mean')
        start_time = time.time()
        if parallel_computation:

            # https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423

            tuples_input = []
            #md = measured_distances
            for i in range(n):
                #tuples_input.append(((md[:, 0][idx_1], md[:, 1][idx_1])
                #                     , (md[:, 0][i], md[:, 1][i])))

                tuples_input.append(((CC[idx_1],pp[idx_1]),(CC[i],pp[i])))

            process_pool = multiprocessing.Pool(32)
            data = tuples_input
            distances = process_pool.starmap(compute_GWD_to_index, data)
            #distances = tqdm.tqdm()
            #for _ in tqdm(process_pool.istarmap(compute_GWD_to_index, data),
             #                  total=len(data)):
              #  pass
            #inputs = zip(param1, param2, param3)
            #with multiprocessing.Pool(10) as pool:
                #distances = pool.starmap(compute_GWD_to_index, tqdm(data, total=len(data)))
            #    distances = list(tqdm(pool.imap(my_function_star, data), total=len(data)))
            #mapped_values = list(tqdm.tqdm(pool.imap_unordered(do_work, range(num_tasks)), total=len(values)))

        else:
            distances = []
            for i in range(0, n):
                print('i: ', i, end=" ")

                gw, log = ot.gromov.entropic_gromov_wasserstein2(
                        CC[idx_1], CC[i],
                        pp[idx_1], pp[i], 'square_loss', epsilon=eps_init, log=True)
                #print(log['gw_dist'])
                distances.append(log['gw_dist'])

                #distances.append(1.0)

            distances = np.asarray(distances)
            distances_for_iter0 = distances
            distances /= distances.sum()
        print("--- %s seconds ---" % (time.time() - start_time))
        print(' ==> Distances computed.')

        # Choose probabilities of choosing next k-1 centers:
        p = np.square(distances)
        p /= p.sum()
        #print(p)

        p[idx_1] = 0
        p /= p.sum()
        #print(p)
        idx_2 = choice(seq, size=1, p=p)[0]
        #print('idx_2: ', idx_2)

        clustering = np.empty(shape=(n,))
        clustering[:] = np.nan

        clustering[idx_1] = 0
        clustering[idx_2] = 1


        #Use dict for cluster centers cc:
        cc = {0: {'dist_m': CC[idx_1], 'weights': pp[idx_1]},
              1: {'dist_m': CC[idx_2], 'weights': pp[idx_2]}}

        # ------------------------------------------------------------------------------------------------------------------
        file_clustering = open(file_name,"w+")
        file_clustering.write('eps_init: ' + str(eps_init))
        file_clustering.write('\n')
        file_clustering.write('eps_update: ' + str(eps_update))
        file_clustering.write('\n')
        file_clustering.write('n_samples_bary: ' + str(n_samples))
        file_clustering.write('\n')


        iter_kmeans = 0
        print(' ==> Starting k-means++-iterations')

        while iter_kmeans < max_iter:

            print('iter: ', iter_kmeans, ' |----------------------------------------------------------------------------------------')
            ##### Zuordnung #####
            # --- Distanzberechnung zu den aktuellen Baryzentren -----------------------------------------------------------
            #print('==> Recalculate distances to each barycenter')
            # Compute for every point, which is not part of a cluster the distance to the cluster centers 1 and 2, take the minimum and assign the point accordingly.
            distances_to_cluster_centers = np.empty(shape=(n, 2))
            distances_to_cluster_centers[:][:] = np.nan

            # Compute distances to all barycenters per data point
            start_time = time.time()
            #print('...starting.')
            for i in range(0, n):

                if (iter_kmeans == 0) and (i == idx_1 or i == idx_2):
                    # Im ersten Iterationsschritt muss für die gewählten Baryzentren nicht der Abstand zu sich selbst berechnet werden
                    continue
                #print('i: ', i)

                for j in range(num_cluster_kmeans):

                    if j == 0 and iter_kmeans == 0:
                        # Distanzen zwischen idx1 und allen anderen Matrizen wurde oben schon berechnet und ist in distances_for_iter0 abgespeichert:
                        distances_to_cluster_centers[i][j] = distances_for_iter0[i]
                    else:
                        # Die Abstände zwischen idx2 und allen anderen Punkten muss noch berechnet werden
                        # Auch für iter_kmeans > 0 müssen die Distanzen neu berechnet werden
                        gw, log = ot.gromov.entropic_gromov_wasserstein2(
                                cc[j]['dist_m'], CC[i],
                                cc[j]['weights'], pp[i], 'square_loss', epsilon=eps_update, log=True, verbose=verbose)#method=method_update
                        distances_to_cluster_centers[i][j] = log['gw_dist']



                #print('Distances to centers: ', distances_to_cluster_centers)


                #print(distances_to_cluster_centers[i])
                #print('argmin: ', distances_to_cluster_centers[i].argmin())
                # --- Cluster Update ---------------------------------------------------------------------------------------
                #Choose minimum distance and set cluster label accordingly:
                clustering[i] = distances_to_cluster_centers[i].argmin()

                #if distances_to_cluster_centers[i].argmin() == 0:
                #    clustering[i] = 0
                #if distances_to_cluster_centers[i].argmin() == 1:
                #    clustering[i] = 1



                #print('==> Update Clustering:' + str(clustering))
                # ----------------------------------------------------------------------------------------------------------
            #  Die einzelnen Punkte sind den Clustern neu zugeordnet

            if iter_kmeans > 0:
                file_clustering = open(file_name, "a")

            file_clustering.write('iter: ' + str(iter_kmeans))
            file_clustering.write('\n')
            #file_clustering.write(str(clustering))
            #file_clustering.write('\n')
            file_clustering.close()

            # Evaluation inside each iteration: ----------------------------------------------------------------------------
            labels_pred = clustering

            # Change clustering labels, if the bigger class is labeled with '1':
            if labels_pred.sum() > labels_pred.shape[0]/2:
                labels_pred = 1 - labels_pred
            a, b, c, d = confusion_matrix(poison_labels, labels_pred).ravel()
            #print('(tn, fp, fn, tp): ', a, b, c, d)

            file_clustering = open(file_name,"a")
            file_clustering.write('(tn, fp, fn, tp): ' + str(a) + ',' + str(b) + ',' + str(c) + ',' + str(d))
            file_clustering.write('\n')
            file_clustering.close()
            # Abbruchkriterium ---------------------------------------------------------------------------------------------
            if iter_kmeans > 0:
                print('Old Clustering', clustering_old)

            print('Clustering: ', clustering)

            if iter_kmeans > 0:
                #Check if clustering has changed in the last iteration:
                if np.equal(clustering_old, clustering).all():
                    print('k-means Clustering didnt change and is being stopped.')
                    break

            clustering_old = deepcopy(clustering)

            #print("--- %s seconds ---" % (time.time() - start_time))


            ##### Aktualisierung #####--------------------------------------------------------------------------------------
            # Recalculate barycenters

            start_time = time.time()
            print('==> Recalculate barycenters per cluster')

            for j in range(num_cluster_kmeans):
                idd = np.where(clustering==id)
                print('idd:', idd[0], end=" ")
                bary = compute_barycenter_from_Cp(CC, pp, clustering=clustering, id=j, n_samples=n_samples, entropic=True)
                p = ot.unif(bary.shape[0])

                cc[j]['dist_m'] = bary
                cc[j]['weights'] = p

            iter_kmeans +=1

            print("--- %s seconds ---" % (time.time() - start_time))

        # --- ENDE kMEans --------------------------------------------------------------------------------------------------

        # Change clustering labels, if the bigger class is labeled with '1':
        if clustering.sum() > clustering.shape[0]/2:
            print('Cluster got flipped')
            clustering = 1 - clustering

        print('final_CLustering: ', clustering)
        # Evaluate clustering:
        a, b, c, d = confusion_matrix(poison_labels, clustering).ravel()
        #specificity = tn / (tn+fp)
        #print(tn, fp, fn, tp)
        # 1650 0 386 427
        print('(tn, fp, fn, tp): ', a, b, c, d)



