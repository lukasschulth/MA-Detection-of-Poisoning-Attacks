from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

with open('clustering_test_iter0.npy', 'rb') as f:

    d = np.load(f)

    print(d.shape)

    #plt.imshow(c)
    #plt.imshow(c.transpose(1, 2, 0))
    #plt.show()



"""
def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False






#d = np.swapaxes(d, 0, 2)
#print(d.shape)
d = np.sum(d, axis=0)
print(d.shape)
print(d.min(), d.max())
plt.imshow(d)
plt.savefig('h1.png')
plt.show()

# Rescaled image:
d = (255.0 / d.max() * (d - d.min())).astype(np.uint8)
plt.imshow(d)
plt.show()
#plt.imshow(c)

"""

#Abspeichern
path = os.getcwd()


im = Image.fromarray(d).convert('RGB')

print(im.size)
im.save("heatmap.jpg", subsampling=0, quality=100)




data = np.random.random((100,100))

#Rescale to 0-255 and convert to uint8
rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)

im = Image.fromarray(rescaled)
im.save('test.png')
