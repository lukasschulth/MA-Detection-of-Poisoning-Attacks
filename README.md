# Detection-of-Poisoning-Attacks

## Thema
Mit dem vorliegenden Code werden verschiedene Arten von Poisoning-Angirffen implementiert und anschließend zwei Detektionsverfahren miteinander verglichen.

## Installation
Die benötigten Python-Bibliotheken finden sich in `requirements.txt` und können wie folgt installiert werden:
```
pip install -r requirements.txt
```
## Benutzung
In `training_lrp` werden die Angriffe implementiert, indem der Datensatz manipuliert und das Training durchgeführt wird. Anschließend werden zu einer verächtigen Klasse die Heatmaps mithilfe der Layer-wise Relevance-Propagation (LRP) berechnet.
Zudem wird eine Detektion basierend auf dem Activation-Clustering ausgeführt.

In `detection_kmeans_gromovwasserstein` werden die erzeugten Heatmaps eingelesen und kMeans-Clustering unter Verwendung der Gromov-Wasserstein-Distanz durchgeführt.



