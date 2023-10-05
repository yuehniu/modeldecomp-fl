import torch.nn as nn
import math


class CNN( nn.Module ):

    def __init__( self, num_classes=62, **kwargs ):
        super( CNN, self ).__init__()
        self.conv1 = nn.Conv2d( 1, 32, kernel_size=5, padding='same' )
        self.relu = nn.ReLU()
        self.pooling1 = nn.MaxPool2d( 2 )
        self.conv2 = nn.Conv2d( 32, 64, kernel_size=5, padding='same' )
        self.pooling2 = nn.MaxPool2d( 2 )
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear( 3136, 2048 )
        # self.fc2 = nn.Linear( 2048, 62 )
        self.fc2 = nn.Linear( 3136, 62 )
        self._initialize_weights()

    def forward( self, x ):
        x = self.relu( self.conv1( x ) )
        x = self.pooling1( x )
        x = self.relu( self.conv2( x ) )
        x = self.pooling2( x )
        x = self.flatten( x )
        # x = self.relu( self.fc1( x ) )
        x = self.fc2( x )
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()