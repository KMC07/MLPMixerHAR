import einops
import torch
import torch.nn as nn
from os.path import join as pjoin

TOKEN_FC_0 = "token_mixing/Dense_0/"
TOKEN_FC_1 = "token_mixing/Dense_1/"
CHANNEL_FC_0 = "channel_mixing/Dense_0/"
CHANNEL_FC_1 = "channel_mixing/Dense_1/"
PRE_NORM = "LayerNorm_0/"
POST_NORM = "LayerNorm_1/"

class MlpBlock(nn.Module):
    '''
    :parameter

    in_layer : int
        Number of inputs connect to the MLP

    hidden_layer : int
        Number of nodes in the hidden layer

    output_layer : int
        Number of output nodes in the output layer

    :var
    fc : nn.Linear
        The first layer of the MLP

    fc2 : nn.Linear
        The second layer of the MLP

    act : nn.GELU
        GELU Activation function
    '''
    def __init__(self, in_layer, hidden_layer, output_layer):
        super().__init__()
        self.fc = nn.Linear(in_layer, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, output_layer)
        self.act = nn.GELU()

    def forward(self, x):
        '''
        :param x: torch.Tensor
                Shape(num_samples, num_channels, num_patches) or Shape(num_samples, num_patches, num_channels)

        :return: torch.Tensor
                Shape(num_samples, num_channels, num_patches) or Shape(num_samples, num_patches, num_channels)
                will be the same as the input
        '''

        x = self.fc(x)
        x = self.act(x)
        out = self.fc2(x)
        return out

def np2torch(weights, input=False):
    """convert the numpy array into a torch tensor."""
    if input:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class MixerBlock(nn.Module):
    '''
    Mixer block that contains two 'layer Norm', 'skip connections' and 'MLP Block'

    :parameter
    num_patches : int
        Number of patches the input will be split to

    patch_dim : int
        This is the dimensions of the patches

    token_dim : int
        This is the dimensions for the hidden layer of the mlp when doing the token mixing

    channel_dim : int
        This is the dimensions for the hidden layer of the mlp when doing the channel mixing

    :var
    token_mixing : nn.Linear
        Token mlp mixer block

    channel_mixing : nn.Linear
        Channel mlp mixer block

    pre_norm : nn.LayerNorm
        Layer normalization

    post_norm : nn.LayerNorm
        Layer normalization
    '''
    def __init__(self, num_patches, patch_dim, token_dim, channel_dim, NoToken, NoChannel):
        super().__init__()
        self.NoToken = NoToken
        self.NoChannel = NoChannel

        if not NoToken:
            self.pre_norm = nn.LayerNorm(patch_dim)
            self.token_mixing = MlpBlock(num_patches, token_dim, num_patches)
        if not NoChannel:
            self.post_norm = nn.LayerNorm(patch_dim)
            self.channel_mixing = MlpBlock(patch_dim, channel_dim, patch_dim)

    def load_pretrained(self, weights, num_block):
        ROOT = f"MixerBlock_{num_block}/"
        with torch.no_grad():
            self.token_mixing.fc.weight.copy_(
                np2torch(weights[pjoin(ROOT, TOKEN_FC_0, "kernel")]).t())
            self.token_mixing.fc.weight.requires_grad = False
            self.token_mixing.fc2.weight.copy_(
                np2torch(weights[pjoin(ROOT, TOKEN_FC_1, "kernel")]).t())
            self.token_mixing.fc2.weight.requires_grad = False
            self.token_mixing.fc.bias.copy_(
                np2torch(weights[pjoin(ROOT, TOKEN_FC_0, "bias")]).t())
            self.token_mixing.fc.bias.requires_grad = False
            self.token_mixing.fc2.bias.copy_(
                np2torch(weights[pjoin(ROOT, TOKEN_FC_1, "bias")]).t())
            self.token_mixing.fc2.bias.requires_grad = False


            self.channel_mixing.fc.weight.copy_(
                np2torch(weights[pjoin(ROOT, CHANNEL_FC_0, "kernel")]).t())
            self.channel_mixing.fc.weight.requires_grad = False
            self.channel_mixing.fc2.weight.copy_(
                np2torch(weights[pjoin(ROOT, CHANNEL_FC_1, "kernel")]).t())
            self.channel_mixing.fc2.weight.requires_grad = False
            self.channel_mixing.fc.bias.copy_(
                np2torch(weights[pjoin(ROOT, CHANNEL_FC_0, "bias")]).t())
            self.channel_mixing.fc.bias.requires_grad = False
            self.channel_mixing.fc2.bias.copy_(
                np2torch(weights[pjoin(ROOT, CHANNEL_FC_1, "bias")]).t())
            self.channel_mixing.fc2.bias.requires_grad = False

            self.pre_norm.weight.copy_(np2torch(weights[pjoin(ROOT, PRE_NORM, "scale")]))
            self.pre_norm.weight.requires_grad = False
            self.pre_norm.bias.copy_(np2torch(weights[pjoin(ROOT, PRE_NORM, "bias")]))
            self.pre_norm.bias.requires_grad = False
            self.post_norm.weight.copy_(np2torch(weights[pjoin(ROOT, POST_NORM, "scale")]))
            self.post_norm.weight.requires_grad = False
            self.post_norm.bias.copy_(np2torch(weights[pjoin(ROOT, POST_NORM, "bias")]))
            self.post_norm.bias.requires_grad = False

    def forward(self, x):
        '''
        :param x: torch.Tensor
                Shape(num_samples, num_patches, patch_dim)

        :return: torch.Tensor
                Shape(num_samples, num_patches, patch_dim)
        '''
        if self.NoToken:
            # channel mixing
            channel = self.post_norm(x)
            channel = self.channel_mixing(channel)

            out = channel
            return out

        elif self.NoChannel:
            # perform the token mixing
            token = self.pre_norm(x)
            token = torch.transpose(token, 1, 2)
            token = self.token_mixing(token).transpose(2, 1)
            out = token
            return out

        else:
            #perform the token mixing
            token = self.pre_norm(x)
            token = torch.transpose(token, 1, 2)
            token = self.token_mixing(token).transpose(2, 1)

            #skip connections
            conn = x + token

            #channel mixing
            channel = self.post_norm(conn)
            channel = self.channel_mixing(channel)

            out = conn + channel
            return out

class MlpMixer(nn.Module):
    """
    Mlp mixer model

    :parameter
    image_height : int
        Height of the input data

    image_width : int
        Width of the input data

    patch_size : int
        Height and width of the patches, where there is no leftover image i.e image size is fully divisible by the patch size

    token_dim : int
        This is the dimensions for the hidden layer of the mlp when doing the token mixing

    channel_dim : int
        This is the dimensions for the hidden layer of the mlp when doing the channel mixing

    num_classes : int
        Number of classes that used for classification

    patch_dim : int
        Dimensionality of the patch embeddings

    num_blocks : int
        The number of mixer blocks

    :var
    patch_embedding : nn.Conv2D
        Converts the image into the different patches and adds embedding

    blocks : nn.ModuleList
        Module list of all the mixer blocks

    classifier_head : nn.Linear
        The classifier head

    head_norm : nn.LayerNorm
        Layer normalization used before the head
    """

    def __init__(self, image_height, image_width, patch_size, token_dim, channel_dim, patch_dim, num_classes, num_blocks, NoToken=False, NoChannel=False, NoRGB=False):
        super().__init__()
        #this is necessary so when the model is saved so its parameters
        self.kwargs = {'image_height': image_height, 'image_width': image_width, 'patch_size': patch_size,
                       'token_dim': token_dim, 'channel_dim': channel_dim, 'patch_dim': patch_dim,
                       'num_classes': num_classes, 'num_blocks': num_blocks}
        self.num_patches = ((image_width // patch_size ) * (image_height // patch_size)) #there should be no remainder
        self.NoRGB = NoRGB
        if not NoRGB:
            self.har_embedding = nn.Conv2d(1, 3, kernel_size=(15), padding="same", stride=1) #this is neccessary to convert the har into 3 dimensions
        self.patch_embedding = nn.Conv2d(1 if NoRGB else 3, patch_dim, padding=1, kernel_size=(patch_size), stride=(patch_size)) #splits images into patches
        self.MixerBlock = nn.ModuleList(
            [
               MixerBlock(num_patches=self.num_patches, patch_dim=patch_dim, token_dim=token_dim, channel_dim=channel_dim, NoToken=NoToken, NoChannel=NoChannel)
                for _ in range(num_blocks)
            ]
        )

        self.head_norm = nn.LayerNorm(patch_dim)
        self.classifer_head = nn.Linear(patch_dim, num_classes)

    def load_pretrained(self, weights):
        with torch.no_grad():
            nn.init.zeros_(self.classifer_head.weight)
            nn.init.zeros_(self.classifer_head.bias)
            self.patch_embedding.weight.copy_(np2torch(weights["stem/kernel"], input=True))
            self.patch_embedding.weight.requires_grad=False
            self.patch_embedding.bias.copy_(np2torch(weights["stem/bias"]))
            self.patch_embedding.bias.requires_grad=False
            self.head_norm.weight.copy_(np2torch(weights["pre_head_layer_norm/scale"]))
            self.head_norm.weight.requires_grad=False
            self.head_norm.bias.copy_(np2torch(weights["pre_head_layer_norm/bias"]))
            self.head_norm.bias.requires_grad=False

            for bname, block in self.MixerBlock.named_children():
                block.load_pretrained(weights, num_block=bname)

    def forward(self, x):
        '''
        :param x: torch.Tensor
            shape(num_smaples, sequence_features, sequence_timeline)

        :return: torch.Tensor
            predicted class of shape(num_samples, num_classes)
        '''
        #add another dimension to the data to represent number of channels
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            #shape(num_smaples, num_channels, sequence_features, sequence_timeline)

        #convert the input into 3 dimensions for the mlp mixer
        if not self.NoRGB:
            x = self.har_embedding(x)

        #split the input into patches and apply the embedding
        patch = self.patch_embedding(x)
        patch = patch.flatten(2)
        patch = patch.transpose(-1, -2)

        #run the mixer blocks
        for block in self.MixerBlock:
            patch = block(patch)

        #apply the head normalization
        out = self.head_norm(patch)
        #apply the global average pooling
        out = out.mean(dim = 1)
        #apply the classifier
        out = self.classifer_head(out)
        return out