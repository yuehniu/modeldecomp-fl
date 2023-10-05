"""Download and create FEMNIST dataset
"""
import os.path

import cv2
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_datasets as tfds

data_dir = './femnist/'
export_train = False
export_test = True

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data( only_digits=False )

n_total_client = 3400
n_per_group = 10

if export_train:
    data_dir = data_dir + 'train/'
    for i in range( n_total_client ):
        print( 'client: ', i )
        grp_id = 'grp' + str( i // n_per_group )
        data_client_i = emnist_train.create_tf_dataset_for_client( emnist_train.client_ids[ i ] )
        data_client_i = tfds.as_numpy( data_client_i )

        for j, data in enumerate( data_client_i ):
            img, label = data[ 'pixels' ], data[ 'label' ]
            img_path = data_dir + '/' + str(grp_id) + '/' + str( label )
            if not os.path.exists( img_path ):
                os.makedirs( img_path )
            img_name = str(i) + '_' + str( j ) +'.jpg'
            cv2.imwrite( img_path + '/' + img_name , img*255 )


if export_test:
    data_dir = data_dir + 'val/'
    for i in range( n_total_client ):
        print( 'client: ', i )
        data_client_i = emnist_test.create_tf_dataset_for_client( emnist_test.client_ids[ i ] )
        data_client_i = tfds.as_numpy( data_client_i )

        for j, data in enumerate( data_client_i ):
            img, label = data[ 'pixels' ], data[ 'label' ]
            img_path = data_dir + '/' + str( label )
            if not os.path.exists( img_path ):
                os.makedirs( img_path )
            img_name = str(i) + '_' + str( j ) +'.jpg'
            cv2.imwrite( img_path + '/' + img_name , img*255 )