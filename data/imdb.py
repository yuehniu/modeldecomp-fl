import re
import torch
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class imdb( object ):
    def __init__( self, data_path ):
        df = pd.read_csv( data_path )
        X, y = df[ 'review' ].values, df[ 'sentiment' ].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( X, y, stratify=y )

    def preprocess_string( self, s ):
        # remove all non-word characters
        s = re.sub( r"[^\w\s]", '', s )
        # remove all runs of whitespaces with no space
        s = re.sub( r"\s+", '', s )
        # replace digits with no space
        s = re.sub( r"\d", '', s )

        return s

    def tockenize( self ):
        word_list = []
        stop_words = set( stopwords.words( 'english' ) )
        for sent in self.x_train:
            for word in sent.lower().split():
                word = self.preprocess_string( word )
                if word not in stop_words and word != '':
                    word_list.append( word )

        corpus = Counter( word_list )
        corpus_ = sorted( corpus, key=corpus.get, reverse=True )[ :1000 ]
        onehot_dict = { w:i+1 for i, w in enumerate( corpus_ ) }

        # tockenize
        final_list_train, final_list_test =[], []
        for sent in self.x_train:
            final_list_train.append(
                [
                    np.int32 ( onehot_dict[ self.preprocess_string( word ) ] ) for word in sent.lower().split()
                    if self.preprocess_string( word ) in onehot_dict.keys()
                ]
            )

        for sent in self.x_test:
            final_list_test.append(
                [
                    np.int32( onehot_dict[ self.preprocess_string( word ) ] ) for word in sent.lower().split()
                    if self.preprocess_string( word ) in onehot_dict.keys()
                ]
            )

        encoded_train = [ 1 if label=='positive' else 0 for label in self.y_train ]
        encoded_test = [ 1 if label=='positive' else 0 for label in self.y_test ]

        x_train = np.array( final_list_train )
        y_train = np.array( encoded_train ).astype( np.int32 )
        x_test = np.array( final_list_test )
        y_test = np.array( encoded_test ).astype( np.int32 )

        return x_train, y_train, x_test, y_test, onehot_dict

    def padding( self, sentences, seq_len ):
        features = np.zeros( ( len( sentences ), seq_len ), dtype=np.int32 )
        for ii, review in enumerate( sentences ):
            if len( review ) != 0:
                features[ ii, -len( review ): ] = np.array( review )[ :seq_len ]

        return features

    def dataset( self ):
        x_train, y_train, x_test, y_test, vocab = self.tockenize()
        x_train_pad, x_test_pad = self.padding( x_train, 500 ), self.padding( x_test, 500 )
        # train_data = TensorDataset( torch.from_numpy( x_train_pad ), torch.from_numpy( y_train ) )
        # test_data = TensorDataset( torch.from_numpy( x_test_pad ), torch.from_numpy( y_test ) )

        # train_loader = DataLoader( train_data, shuffle=True, batch_size=batch_size, drop_last=True )
        # test_loader = DataLoader( test_data, shuffle=True, batch_size=batch_size, drop_last=True )

        return x_train_pad, y_train, x_test_pad, y_test, vocab
