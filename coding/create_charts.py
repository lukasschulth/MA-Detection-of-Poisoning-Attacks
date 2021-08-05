import matplotlib.pyplot as plt

import numpy as np

ls = [0.71666667,
0.78333333,
0.80266667,
0.99333333,
0.9469697,
0,
0.96,
0.97555556,
0.98444444,
1.,
1.,
0.25714286,
0.98985507,
0.86111111,
0.94444444,
0.99047619,
0.91333333,
0.51666667,
0.84871795,
0.86666667,
0.52222222,
0.57777778,
0.75,
0.67333333,
0.95555556,
0.725,
0.98888889,
0.88333333,
0.45333333,
0.82222222,
0.81333333,
1.,
0.98333333,
0.4952381,
0.26666667,
0.92820513,
0.63333333,
0.81666667,
0.77681159,
0.33333333,
0.74444444,
0.91666667,
0.83333333]


ls2 = [0.51666667, 0.43194444, 0.64,0.93555556, 0.81818182,0,
	0.92,       0.93333333, 0.95333333, 0.52916667, 0.74545455, 0.08809524,
	0.50144928, 0.34166667, 0.27407407, 0.36190476, 0.34,       0.15555556,
	0.5,        0.6,        0.33333333, 0.16666667, 0.45833333, 0.34666667,
	0.83333333, 0.15833333, 0.36111111, 0.7,        0.27333333, 0.38888889,
	0.8,        0.93333333, 0.5,        0.2047619,  0.00833333, 0.63846154,
	0.1,        0.28333333, 0.29130435, 0.33333333, 0.42222222, 0.36666667,
	0.74444444]

ls3 = [0.05,       0.09444444, 0.19466667, 0.32888889, 0.22727273,  0,
 0.1,        0.21555556, 0.16444444, 0.05416667, 0.05757576, 0.01904762,
 0.12463768, 0.,         0.01111111, 0.,         0.,         0.05277778,
 0.07692308, 0.,         0.05555556, 0.,         0.04166667, 0.00666667,
 0.02222222, 0.01875,    0.,         0.06666667, 0.04666667, 0.,
 0.01333333, 0.02592593, 0.,         0.,         0.,         0.11282051,
 0.,         0.,         0.06956522, 0.33333333, 0.05555556, 0.,
 0.11111111]

npls = 100 * np.asarray(ls)
print('mAER_255: ', sum(ls)/42)
print('min_255: ', np.sort(npls)[1])
print('max2_255 ', npls.max())

npls2 = 100 * np.asarray(ls2)
print('mAER_64: ', sum(ls2)/42)
print('min_64: ', np.sort(npls2)[1])
print('max2_64 ', npls2.max())

x = list(range(1,44))


width = 0.27

fig = plt.figure()
ax = fig.add_subplot(111)

N = 43
yy = np.arange(1,N+1)
rects1 = ax.bar(yy-0.5*width, npls, width, color='b')
rects2 = ax.bar(yy+0.5*width, npls2, width, color='c')


ax.set_ylabel('AER (in %)')
ax.set_xlabel('Klassen im Datensatz')
ax.legend((rects1[0], rects2[0]), ('amp=255', 'amp=64'))

plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.show()