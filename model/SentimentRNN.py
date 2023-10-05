import torch
import torch.nn as nn

device = torch.device( 'cuda' )


class SentimentRNN( nn.Module ):
    def __init__(
            self, vocab_size, n_lstm_layers=2, dim_hidden=256, dim_embed=64, dim_out=1, drop_prob=0.5
    ):
        super( SentimentRNN, self ).__init__()

        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.dim_embed = dim_embed

        self.n_lstm_layer = n_lstm_layers
        self.vocab_size = vocab_size

        # - define module
        self.embedding = nn.Embedding( vocab_size, dim_embed )
        self.lstm = nn.LSTM(
            input_size=dim_embed, hidden_size=dim_hidden,
            num_layers=n_lstm_layers, batch_first=True
        )
        # self.dropout = nn.Dropout( drop_prob )
        self.fc = nn.Linear( dim_hidden, dim_out )
        self.sig = nn.Sigmoid()

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
        h0 = torch.zeros( self.n_lstm_layer, batch_size, self.dim_hidden ).to( device )
        c0 = torch.zeros( self.n_lstm_layer, batch_size, self.dim_hidden ).to( device )
        hidden = ( h0, c0 )

        return hidden
