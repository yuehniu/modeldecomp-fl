"""
This script visualize convolution outputs using orthogonal kernels.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}


# - create a input tensor for Torch conv module
img = cv2.imread( './images/cat.jpeg' )[ :, :, [ 2, 1, 0 ] ]
t_img = torch.from_numpy( img )
t_img = torch.permute( t_img, [ 2, 0, 1 ] ).unsqueeze( 0 ) / 255.0


# - define a conv layer
s1, s2, s3 = 1.0, 0.7, 0.5
n_ichnl, n_ochnl, k = 3, 3, 3
m_W = np.random.randn( n_ochnl, n_ichnl*k*k ).astype( np.float32 )
m_U, v_s, m_V = np.linalg.svd( m_W, full_matrices=False )
m_W0 = v_s[ 0 ] * np.matmul( m_U[ :, [0] ], m_V[ [0], : ] )
m_W1 = s2 * v_s[ 1 ] * np.matmul( m_U[ :, [1] ], m_V[ [1], : ] )
m_W2 = s3 * v_s[ 2 ] * np.matmul( m_U[ :, [2] ], m_V[ [2], : ] )
t_W0 = torch.from_numpy( m_W0 ).view( n_ochnl, n_ichnl, k, k )
t_W1 = torch.from_numpy( m_W1 ).view( n_ochnl, n_ichnl, k, k )
t_W2 = torch.from_numpy( m_W2 ).view( n_ochnl, n_ichnl, k, k )

# - apply conv
t_output0 = F.conv2d( t_img, t_W0 )
t_output1 = F.conv2d( t_img, t_W1 )
t_output2 = F.conv2d( t_img, t_W2 )
# t_output0 = F.relu( t_output0 )
# t_output1 = F.relu( t_output1 )
# t_output2 = F.relu( t_output2 )
m_output0 = t_output0.squeeze().permute( [ 1, 2, 0 ] ).numpy()
m_output1 = t_output1.squeeze().permute( [ 1, 2, 0 ] ).numpy()
m_output2 = t_output2.squeeze().permute( [ 1, 2, 0 ] ).numpy()

# - visualize
min_0, range_0 = np.min(m_output0), np.max( m_output0 ) - np.min( m_output0 )
min_1, range_1 = np.min(m_output1), np.max( m_output1 ) - np.min( m_output1 )
min_2, range_2 = np.min(m_output2), np.max( m_output2 ) - np.min( m_output2 )
i, j, k = 0, 7, 15
corr = np.matmul( m_output0.reshape( 1, -1 ),  m_output1.reshape( -1, 1 ) )
print( 'correlation', corr )

plt.figure( 1, figsize=[ 8, 2 ], facecolor=[0.95, 0.95, 0.95] )
plt.subplot( 1, 4, 1 )
plt.imshow( img )
plt.axis( 'off' )
plt.title( 'Inputs', **csfont )
plt.subplot( 1, 4, 2 )
plt.imshow( ( m_output0 - min_0 ) / range_0 )
plt.axis( 'off' )
plt.title( 'Kernel 1', **csfont )
plt.subplot( 1, 4, 3 )
plt.imshow( ( m_output1 - min_1 ) / range_1 * s2 )
plt.axis( 'off' )
plt.title( 'Kernel 2', **csfont )
plt.subplot( 1, 4, 4 )
plt.imshow( ( m_output2 - min_2 ) / range_2 * s3 )
plt.axis( 'off' )
plt.title( 'Kernel 3', **csfont )

plt.show()