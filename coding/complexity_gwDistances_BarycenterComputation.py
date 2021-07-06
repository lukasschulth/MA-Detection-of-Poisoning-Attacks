import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

x = [5, 25, 50, 100]
time_distances = [379, 1861, 1956, 4670]
time_barycenters = [725, 2444, 5670,11512 ]

td = np.asarray(time_distances)
td = td/3600 # Umrechnung von Sekunden in Stunden

tb = np.asarray(time_barycenters)
tb = tb/3600 # Umrechnung von Sekunden in Stunden

plt.plot(x, td, '*-')
plt.plot(x, tb, '*-')
plt.title('Benötigte Zeit für die Berechnung der Wasserstein-Distanzen und Baryzentren')
plt.legend(['Wasserstein-Distanzen','Wasserstein-Baryzentren'])
plt.xlabel('Anzahl an betrachteten Bildern')
plt.ylabel('Zeit in h')
plt.show()

clustering = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,
 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0,
 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
 1, 1, 0, 1]
cl = np.asarray(clustering)
cl = 1-cl
poisonLabels = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]

a,b,c,d = confusion_matrix(poisonLabels, cl).ravel()
#specificity = tn / (tn+fp)
#print(tn, fp, fn, tp)
# 1650 0 386 427
print(a, b, c, d)
