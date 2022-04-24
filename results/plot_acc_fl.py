import matplotlib.pyplot as plt
import numpy as np


keep = np.array(
    [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 ]
)

# random drop orthogonal channels in U
acc_orth_dropout_random = np.array(
    [ None, 89.18, 90.27, 91.12, 91.67, 92.2 ]
)
acc_orth_dropout_random = np.array(
    [ None, 92.0, 92.578, 93.379, 93.359, 93.369 ]
)

# random drop orthogonal channels in U and V
acc_orth_dropout_random_UV = [
    None, None, None, 89.533, 91.677, 93.369
]

# only keep top-k principal channels
acc_orth_dropout_fedhm = np.array(
    [ None, 89.944, 91.17, 91.89, 92.167, 93.399 ]
)
acc_orth_dropout_fedhm = np.array(
    [ None, 84.265, 88.933, 89.774, 91.757, 93.399 ]
)
acc_orth_dropout = np.array(
    [  ]
)
acc_reg_dropout = np.array(
    [  ]
)
acc_fedhm = np.array(
    [  ]
)

plt.plot( keep, acc_orth_dropout_random, 's--', label='Orth Dropout' )
# plt.plot( keep, acc_reg_dropout, 'rs--', label='Reg Dropout' )
plt.plot( keep, acc_orth_dropout_fedhm, 'ks--', label='FedHM' )

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



