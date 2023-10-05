# profile the kernel sampling process
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams[ "font.family" ] = "Times New Roman"

from utils.meter import cal_entropy

file = 'results/sampling_stats.npy'

data = np.load( file, allow_pickle=True )
keys = data.item()

offset = 0.0
legends = [
    'Block1-1', 'Block1-2', 'Block2-1', 'Block2-2',
    'Block3-1', 'Block3-2', 'Block4-1', 'Block4-2',
    'Block5-1', 'Block5-2', 'Block6-1', 'Block6-2',
    'Block7-1', 'Block7-2', 'Block8-1', 'Block8-2'
]
fontsize = 15
i = 0
for key in keys:
    if '64' in key:
        plt.figure( 1 )
    elif '128' in key:
        plt.figure( 2 )
    elif '256' in key:
        plt.figure( 3 )
    elif '512' in key:
        plt.figure( 4 )

    data_i = data.item()[ key ]
    x = np.arange( len( data_i ) ) + offset
    # plt.bar( x, data_i, width=1.0, label=legends[ i ], alpha=1.0 )
    plt.plot( x, data_i / 100, '--', label=legends[ i ] )
    # plt.scatter( x, data_i, label=legends[ i ] )

    offset += 0.0
    i += 1

    plt.legend( ncol=2, fontsize=fontsize )
    plt.grid( axis='y', linestyle='--' )
    plt.xticks( fontsize=15 )
    plt.yticks( [ 0, 4, 8, 12, 16, 20], fontsize=15 )
plt.show()