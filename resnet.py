import torch.nn as nn


###############################################
###############################################
#https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
###############################################
###############################################


class ResidualBlock1(nn.Module):
    """
    Block used for a ResNet model where every layer consists of one convolutional layer.

    Attributes:
    -----------
        self.conv1 : first convolutional layer with input dimension, output dimension, kernel size =3,
        variable stride and padding =1.

        self.downsample : either none or is a convolutional layer which resamples the input of conv1, X, to
        match the output dimensions of conv2 such that the dimensional output of downsample(X) = conv2(conv1(X)).
        
        self.relu : relu activation function

        self.out_channels : sets the size of the output channel of the convolutional layer.

    Methods:
    --------
        forward : completes a forward pass through the model.

    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        """
        Initialize the 2 layer deep cnn block
    
        Parameters:
        -----------
            in_channels : size of the input channel.

            out_channels : size of the output channel.

            stride : length of the stride for the kernel.

            downsample : amount to alter the input to the block such that it can be
            added directly to the output (the input to the block will remain unaltered).

        """
        super(ResidualBlock1, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            x : input frames to be passed through the block.

        Returns:
        --------
            out : result of passing the input through conv1 and conv2 added with downsample(x) or 
            if downsample =none, added with x directly.

        """
        residual = x
        out = self.conv1(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class ResidualBlock2(nn.Module):
    """
    Block used for a ResNet model where every layer consists of two 
    near identical convolutional layers.

    Attributes:
    -----------
        self.conv1 : first convolutional layer with input dimension, output dimension, kernel size =3,
        variable stride and padding =1.
        
        self.conv2 : second convolutional layer with input dimension, output dimension, kernel size =3,
        stride =1 and padding =1.

        self.downsample : either none or is a convolutional layer which resamples the input of conv1, X, to
        match the output dimensions of conv2 such that the dimensional output of downsample(X) = conv2(conv1(X)).
        
        self.relu : relu activation function

        self.out_channels : sets the size of the output channel of the convolutional layer.

    Methods:
    --------
        forward : completes a forward pass through the model.

    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        """
        Initialize the 2 layer deep cnn block
    
        Parameters:
        -----------
            in_channels : size of the input channel.

            out_channels : size of the output channel.

            stride : length of the stride for the kernel.

            downsample : amount to alter the input to the block such that it can be
            added directly to the output (the input to the block will remain unaltered).

        """
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            x : input frames to be passed through the block.

        Returns:
        --------
            out : result of passing the input through conv1 and conv2 added with downsample(x) or 
            if downsample =none, added with x directly.

        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    


class ResidualBlock3(nn.Module):
    """
    Block used for a ResNet model where every convolutional layer consists of two dimensionally similar convolutional layers
    followed by a third convolutional layer which is 4* the output dimension of the previous convolutional layer.

    Attributes:
    -----------
        self.conv1 : first convolutional layer with input dimension, output dimension, kernel size,
        stride and padding.
        
        self.conv2 : second convolutional layer with input dimension, output dimension, kernel size,
        stride and padding.

        self.conv3 : third convolutional layer with input dimension, output dimension, kernel size, 
        stride and padding.

        self.downsample : either none or is a convolutional layer which resamples the input of conv1, X, to
        match the output dimensions of conv2 such that the dimensional output of downsample(X) = conv3(conv2(conv1(X))).
        
        self.relu : relu activation function.

        self.out_channels : sets the size of the output dimension of each convolutional layer.

    Methods:
    --------
        forward : completes a forward pass through the model.

    """
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        """
        Initialize the 3 layer deep cnn block.
    
        Parameters:
        -----------
            in_channels : size of the input channel.

            out_channels : size of the output channel.

            stride : length of the stride for the kernel.

            downsample : amount to alter the input to the block such that it can be
            added directly to the output (the input to the block will remain unaltered).

        """
        super(ResidualBlock3, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, 4*out_channels, kernel_size = 1, stride = 1, padding = 0),
                        nn.BatchNorm2d(4*out_channels))
    
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            x : input frames to be passed through the block.

        Returns:
        --------
            out : result of passing the input through conv1 and conv2 added with downsample(x) or 
            if downsample =none, added with x directly.

        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    


class ResNet_4layer(nn.Module):
    """
    ResNet model which has 4 distinct layers. Each layer must have the same number of output channels.

    Attributes:
    -----------
        self.inplanes : size of the first input dimension
        
        self.conv1 : first convolutional layer with input dimension, output dimension, kernel size,
        stride and padding. Followed by a batch norm and a relu activation function.

        self.maxpool : max pooling of the output of the first convolutional layer. Maxpool has
        kernel size, stride and padding.

        self.layerX : layer of the resnet with block, convolutional layer dimension,
        number of convolutional layers to be in the X layer and stride to be used in
        certain convolutional layers for X layer.

        self.avgpool : average pooling of the output of the final convolutional layer. Average
        pool has kernel size and stride.

        self.fc : final layer of the network which is a fully connected network where the output
        size will be the number of classes the input can be classified into.
        


    Methods:
    --------
        make_layer : create a single layer which will consist of a certain number of 
        passes through the given block.
        
        forward : completes a forward pass through the model.

    """
    def __init__(self, block,layers , num_classes = 10):
        """
        Initialize the 4 layer deep resnet.
    
        Parameters:
        -----------
            block : type of block to be used with this model.

            layers: number of blocks for each layer

            num_classes : number of classes the output can be classified into.

        """
        super(ResNet_4layer, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            block : the type of block to be used in the model.

            planes : size of the input dimension.

            blocks : number of blocks to be created.

            stride : size of the stride for certain convolutional layers.

        Returns:
        --------
            layers : result of sequencing all the layers constructed.

        """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x, length):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            x : input frames to be passed through the block.

        Returns:
        --------
            out : result of passing input frame through all layers.

        """
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    


class ResNet_4layer_3deep(nn.Module):
    """
    ResNet model which has 4 distinct layers. each layer must consit of 3 conv nets. 
    The first 2 conv nets must have the same number of output channels while the third conv net 
    should have 4x the number of output channels compared to the first 2.

    Attributes:
    -----------
        self.inplanes : size of the first input dimension
        
        self.conv1 : first convolutional layer with input dimension, output dimension, kernel size,
        stride and padding. Followed by a batch norm and a relu activation function.

        self.maxpool : max pooling of the output of the first convolutional layer. Maxpool has
        kernel size, stride and padding.

        self.layerX : layer of the resnet with block, convolutional layer dimension,
        number of convolutional layers to be in the X layer and stride to be used in
        certain convolutional layers for X layer.

        self.avgpool : average pooling of the output of the final convolutional layer. Average
        pool has kernel size and stride.

        self.fc : final layer of the network which is a fully connected network where the output
        size will be the number of classes the input can be classified into.
        


    Methods:
    --------
        make_layer : create a single layer which will consist of a certain number of 
        passes through the given block.
        
        forward : completes a forward pass through the model.

    """
    def __init__(self, block, layers, num_classes = 10):
        """
        Initialize the 4 layer deep resnet.
    
        Parameters:
        -----------
            block : type of block to be used with this model.

            layers: number of blocks for each layer

            num_classes : number of classes the output can be classified into.

        """
        super(ResNet_4layer_3deep, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            block : the type of block to be used in the model.

            planes : size of the input dimension.

            blocks : number of blocks to be created.

            stride : size of the stride for certain convolutional layers.

        Returns:
        --------
            layers : result of sequencing all the layers constructed.

        """
        
        downsample = None
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, 4*planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(4*planes),
        )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = 4*planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x, lengths):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            x : input frames to be passed through the block.

        Returns:
        --------
            out : result of passing input frame through all layers.

        """
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    


class ResNet_2layer(nn.Module):
    """
    Block used for a ResNet model which has 2 distinct layers. Each layer must have the same number of output channels.

    Attributes:
    -----------
        self.inplanes : size of the first input dimension
        
        self.conv1 : first convolutional layer with input dimension, output dimension, kernel size,
        stride and padding. Followed by a batch norm and a relu activation function.

        self.maxpool : max pooling of the output of the first convolutional layer. Maxpool has
        kernel size, stride and padding.

        self.layerX : layer of the resnet with block, convolutional layer dimension,
        number of convolutional layers to be in the X layer and stride to be used in
        certain convolutional layers for X layer.

        self.avgpool : average pooling of the output of the final convolutional layer. Average
        pool has kernel size and stride.

        self.fc : final layer of the network which is a fully connected network where the output
        size will be the number of classes the input can be classified into.
        


    Methods:
    --------
        make_layer : create a single layer which will consist of a certain number of 
        passes through the given block.
        
        forward : completes a forward pass through the model.

    """
    def __init__(self, block, layers, num_classes = 10):
        """
        Initialize the 2 layer deep resnet.
    
        Parameters:
        -----------
            block : type of block to be used with this model.

            layers: number of blocks for each layer

            num_classes : number of classes the output can be classified into.

        """
        super(ResNet_2layer, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size=28, stride=1)
        self.fc = nn.Linear(128, num_classes)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            block : the type of block to be used in the model.

            planes : size of the input dimension.

            blocks : number of blocks to be created.

            stride : size of the stride for certain convolutional layers.

        Returns:
        --------
            layers : result of sequencing all the layers constructed.

        """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x, lengths):
        """
        Computes the forward pass through the model.

        Parameters:
        -----------
            x : input frames to be passed through the block.

        Returns:
        --------
            out : result of passing input frame through all layers.

        """
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer0(out)
        out = self.layer1(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out