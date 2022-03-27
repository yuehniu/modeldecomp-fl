import matplotlib.pyplot as plt
import numpy as np


keep = np.array(
    [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 ]
)
acc_orth_dropout_iter_random = np.array(
    [ None, 88.57, 92.81, 93.89, 94.63, 95.14]
)
acc_orth_dropout_iter_random = np.array(
    [ None, 91.22, 93.88, 94.83, 95.23, 95.14 ]
)
acc_orth_dropout_iter = np.array(
    [ None, 85.06, 92.82, 94.27, 95.04, 95.14 ]
)
acc_orth_dropout = np.array(
    [ 92.99, 94.26, 94.782, 94.862, 94.882, 95.142 ]
)
acc_reg_dropout = np.array(
    [ 66.93, 77.26, 86.76, 92.41, 94.75,  95.13 ]
)
acc_fedhm = np.array(
    [ None, 72.26, 85.64, 89.66, 91.33, 92.6 ]
)

plt.plot( keep, acc_orth_dropout_iter_random, 's--', label='Orth Dropout' )
plt.plot( keep, acc_reg_dropout, 'rs--', label='Reg Dropout' )
plt.plot( keep, acc_orth_dropout_iter, 'ks--', label='FedHM' )

plt.xlabel( 'keep ratio' )
plt.ylabel( 'Top-1 accuracy' )
plt.xlim( [ 0.2, 1 ] )
plt.ylim( [ 70, 96 ] )
plt.xticks(
    [0.2, 0.4, 0.6, 0.8, 1.0],
    ['0.2', '0.4', '0.6', '0.8', '1.0' ]
)

plt.legend( loc='lower right' )

plt.show()



