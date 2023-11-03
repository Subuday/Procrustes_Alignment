import numpy as np
import matplotlib.pyplot as plt

A1 = np.loadtxt('hand1.dat', delimiter=',')
A2 = np.loadtxt('hand2.dat', delimiter=',')

M = np.dot(A1.T, A2)
U, S, V = np.linalg.svd(M)

R = np.dot(U, V)
A2_rotated = np.dot(A2, R.T)

plt.plot(A1[:, 0], A1[:, 1], color='red', label='Hand 1')
plt.plot(A2[:, 0], A2[:, 1], color='blue', label='Hand 2')
plt.plot(A2_rotated[:, 0], A2_rotated[:, 1], color='green', label='Hand 2 Rotated')
plt.title('Hand Shapes')
plt.legend()
plt.show()

