import torch
import copy
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor

class LSTM_Orth( torch.nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super( LSTM_Orth, self ).__init__()
        self.input_size, self.hidden_size = input_size, hidden_size

        # input gate
        self.ii_gate = torch.nn.Linear( input_size, hidden_size )
        self.hi_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.i_act = torch.nn.Sigmoid()

        # forget gate
        self.if_gate = torch.nn.Linear( input_size, hidden_size )
        self.hf_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.f_act = torch.nn.Sigmoid()

        # cell memory
        self.ig_gate = torch.nn.Linear( input_size, hidden_size )
        self.hg_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.g_act = torch.nn.Tanh()

        # out gate
        self.io_gate = torch.nn.Linear( input_size, hidden_size )
        self.ho_gate = torch.nn.Linear( hidden_size, hidden_size )
        self.o_act = torch.nn.Sigmoid()

        self.act = torch.nn.Tanh()

    def input_gate( self, x, h ):
        x = self.ii_gate( x )
        h = self.hi_gate( h )
        return self.i_act( x + h )

    def forget_gate( self, x, h):
        x = self.if_gate( x )
        h = self.hf_gate( h )
        return self.f_act( x + h )

    def cell_mem( self, i, f, x, h, c_prev ):
        x = self.ig_gate( x )
        h = self.hg_gate( h )

        k = self.g_act( x + h )
        g = k * i

        c = f * c_prev

        c_next = g + c

        return c_next

    def out_gate(self, x, h ):
        x = self.io_gate( x )
        h = self.ho_gate( h )
        return self.o_act( x + h )

    def forward( self, x, tuple_in: tuple[ Tensor, Tensor ] ):
        ( h, c_prev ) = tuple_in

        i = self.input_gate( x, h )

        f = self.forget_gate( x, h )

        c_next = self.cell_mem( i, f, x, h, c_prev )

        o = self.out_gate( x, h )

        h_next = o * self.act( c_next )

        return h_next, c_next


class LSTMs_Orth( torch.nn.Module ):
    def __init__( self, m, keep ):
        super( LSTMs_Orth, self ).__init__()
        self.input_size, self.hidden_size = m.input_size, m.hidden_size
        self.num_layers = m.num_layers

        device = torch.device( 'cuda' )
        self.LSTM1 = LSTM_Orth( self.input_size, self.hidden_size )
        self.LSTM2 = LSTM_Orth( self.hidden_size, self.hidden_size )
        # self.lstm = []
        # self.lstm.append( LSTM_Orth( self.input_size, self.hidden_size ).to( device ) )
        # for l in range( 1, self.num_layers ):
        #     self.lstm.append( LSTM_Orth( self.hidden_size, self.hidden_size ).to( device ) )

    def forward( self, x, hidden_in ):
        hidden_out = []
        lstm_out = []
        ( h1_i, c1_i ) = hidden_in[ 0 ]
        ( h2_i, c2_i ) = hidden_in[ 1 ]
        for i in range( x.size( 1 ) ):
            h1_i, c1_i = self.LSTM1( x[ :, i, : ], ( h1_i, c1_i ) )
            h2_i, c2_i = self.LSTM2( h1_i, ( h2_i, c2_i ) )
            lstm_out += [ h2_i ]

        lstm_out = torch.stack( lstm_out )
        lstm_out = torch.transpose( lstm_out, 0, 1 )
        hidden_out.append( ( h1_i, c1_i ) )
        hidden_out.append( ( h2_i, c2_i ) )

        return lstm_out, hidden_out


class RNN_Orth( torch.nn.Module ):
    def __init__( self, model, keep ):
        super( RNN_Orth, self ).__init__()
        self.n_lstm_layer = model.n_lstm_layer
        self.dim_hidden, self.dim_embed = model.dim_hidden, model.dim_embed
        self.embedding, self.lstm, self.fc, self.sig = None, None, None, None
        for m in model.children():
            if isinstance( m, torch.nn.Embedding ):
                self.embedding = copy.deepcopy( m )
            elif isinstance( m, torch.nn.LSTM ):
                self.lstm = LSTMs_Orth( m, keep )
            elif isinstance( m, torch.nn.Linear ):
                self.fc = copy.deepcopy( m )
            elif isinstance( m, torch.nn.Sigmoid ):
                self.sig = copy.deepcopy( m )

    def forward( self, x, hidden ):
        batch_size = x.size( 0 )
        embeds = self.embedding( x )
        lstm_out, hidden = self.lstm( embeds, hidden )
        lstm_out = lstm_out.contiguous().view( -1, self.dim_hidden )
        out = self.fc( lstm_out )
        sig_out = self.sig( out )
        sig_out = sig_out.view( batch_size, -1 )
        sig_out = sig_out[ :, -1 ]

        return sig_out, hidden

    def init_hidden( self, batch_size ):
        device = torch.device( 'cuda' )
        hidden = []
        for i in range( self.n_lstm_layer ):
            h0 = torch.zeros( batch_size, self.dim_hidden ).to( device )
            c0 = torch.zeros( batch_size, self.dim_hidden ).to( device )
            hidden.append( ( h0, c0 ) )

        return hidden
