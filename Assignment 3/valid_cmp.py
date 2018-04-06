import numpy as np
import matplotlib.pyplot as plt

x_axis = [x+1 for x in range(500)]
h_100_valid = np.load("H_100_valid.npy")
h_500_valid = np.load("H_500_valid.npy")
h_1000_valid = np.load("H_1000_valid.npy")

fig_loss_cmp = plt.figure(1)
plt.title = 'Validation Loss vs. Epoch'
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.plot(x_axis, h_100_valid, '-', label=('100 Hidden Units'))
plt.plot(x_axis, h_500_valid, '-', label=('500 Hidden Units'))
plt.plot(x_axis, h_1000_valid, '-', label=('1000 Hidden Units'))

plt.legend(loc='best')
fig_loss_cmp.savefig("H_valid_loss_cmp.png")
plt.show()
