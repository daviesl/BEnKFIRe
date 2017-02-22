import numpy as np
import matplotlib.pyplot as plt

h = np.loadtxt('makehist.txt')
plt.hist(h)
plt.show()
