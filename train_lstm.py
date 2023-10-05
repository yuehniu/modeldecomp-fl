"""FL train a LSTM model
"""
import torch
import torch.nn as nn
from data.imdb import imdb
from model.SentimentRNN import SentimentRNN

n_lstm_layer = 2
data_path = 'data/imdb/imdb.csv'
dim_embed, dim_out, dim_hidden = 64, 1, 256
device = torch.device( 'cuda' )

# - dataset
batch_size = 32
imdb_data = imdb( data_path )
train_loader, test_loader, vocab = imdb_data.loader( batch_size )
vocab_size = len( vocab ) + 1

# - model
model = SentimentRNN( n_lstm_layer, vocab_size, dim_hidden, dim_embed, dim_out )
model.to( device )

# training
lr = 0.1
clip = 5
epochs = 5
criterion = nn.BCELoss()
optimizer = torch.optim.SGD( model.parameters(), lr=lr, momentum=0.9 )

def acc( pred, label ):
    pred = torch.round( pred.squeeze() )
    return torch.sum( pred == label.squeeze() ).item()

for epoch in range( epochs ):
    model.train()
    h = model.init_hidden( batch_size )
    train_acc, train_loss = 0.0, 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to( device ), labels.to( device )
        h = tuple( [ each.data for each in h ] )

        model.zero_grad()
        output, h = model( inputs, h )

        loss = criterion( output.squeeze(), labels.float() )
        loss.backward()
        train_loss += loss

        acc_i = acc( output, labels )
        train_acc += acc_i

        nn.utils.clip_grad_norm_( model.parameters(), clip )
        optimizer.step()

    val_h = model.init_hidden( batch_size )
    val_acc, val_loss = 0.0, 0.0
    model.eval()
    for inputs, labels in test_loader:
        val_h = tuple( [ each.data for each in val_h ] )
        inputs, labels = inputs.to( device ), labels.to( device )
        output, val_h = model( inputs, val_h )

        loss_i = criterion( output.squeeze(), labels.float() )
        acc_i = acc( output, labels )
        val_loss += loss_i
        val_acc += acc_i

    train_loss /= len( train_loader )
    train_acc /= len( train_loader.dataset )
    val_loss /= len( test_loader )
    val_acc /= len( test_loader.dataset )
    print( f'Epoch {epoch + 1}' )
    print( f'train loss: {train_loss} val_loss: {val_loss}' )
    print( f'train acc: {train_acc} val acc: {val_acc}' )