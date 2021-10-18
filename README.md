# Detection-of-Poisoning-Attacks

## Thema
Mit dem vorliegenden Code werden verschiedene Arten von Poisoning-Angriffen implementiert und anschließend zwei Detektionsverfahren (Activation-Clustering(AC) und Heatmap-Clustering(HC)) miteinander verglichen.

## Installation
Die benötigten Python-Bibliotheken finden sich in `requirements.txt` und können wie folgt installiert werden:
```
pip install -r requirements.txt
```
## Benutzung
In `coding/training_lrp` werden die Angriffe implementiert, indem der Datensatz manipuliert und das Training durchgeführt wird. Anschließend werden zu einer verächtigen Klasse die Heatmaps mithilfe der Layer-wise Relevance-Propagation (LRP) berechnet.
Zudem wird eine Detektion basierend auf dem Activation-Clustering ausgeführt.

In `coding/detection_kmeans_gromovwasserstein` werden die erzeugten Heatmaps eingelesen und ein kMeans-Clustering unter Verwendung der Gromov-Wasserstein-Distanz durchgeführt.



