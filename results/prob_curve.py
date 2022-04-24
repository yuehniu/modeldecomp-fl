import torch
import numpy as np
import matplotlib.pyplot as plt

s = np.arange( 11, 0, -1 )
s2 = s ** 3
s_norm = s / s.sum()
s2_norm = s2 / s2.sum()
t_s = torch.from_numpy( s.astype( np.float ) )
s_softmax_t1 = torch.nn.functional.softmax( t_s ).numpy()
s_softmax_t2 = torch.nn.functional.softmax( t_s / 2 ).numpy()


plt.plot( s_norm, label='normalized s' )
plt.plot( s2_norm, label='normalized s2' )
plt.plot( s_softmax_t1, label='softmax T1' )
plt.plot( s_softmax_t2, label='softmax T2' )

plt.legend()

plt.show()