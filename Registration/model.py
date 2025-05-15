import inspect
import functools
from typing import Tuple, Union, Type, List

import pydoc
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn
from torch.distributions.normal import Normal
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

default_features = [
    32,
    64,
    128,
    256,
    320,
    320
]
default_kernel_sizes = [[3, 3, 3]] * 6
default_strides = [
    [
        1,
        1,
        1
    ],
    [
        2,
        2,
        2
    ],
    [
        2,
        2,
        2
    ],
    [
        2,
        2,
        2
    ],
    [
        2,
        2,
        2
    ],
    [
        2,
        2,
        2
    ]
]

def init_model(config):
    architecture_kwargs = {"n_stages": 6,
                           "features_per_stage": default_features,
                           "conv_op": "torch.nn.modules.conv.Conv3d",
                           "kernel_sizes": default_kernel_sizes,
                           "strides": default_strides,
                           "n_conv_per_stage": [
                               2,
                               2,
                               2,
                               2,
                               2,
                               2
                           ],
                           "n_conv_per_stage_decoder": [
                               2,
                               2,
                               2,
                               2,
                               2
                           ],
                           "conv_bias": True,
                           "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                           "norm_op_kwargs": {
                               "eps": 1e-05,
                               "affine": True
                           },
                           "nonlin": "torch.nn.LeakyReLU",
                           "nonlin_kwargs": {
                               "inplace": True
                           }
                           }
    arch_kwargs_req_import = [
        "conv_op",
        "norm_op",
        "nonlin"
    ]
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    # Init model
    model = VxmDense(
        inshape=config.imgsize,
        input_channels=config.in_channels,
        **architecture_kwargs
    )
    return model


def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if argspec.defaults:
            for attr, val in zip(reversed(argspec.args), reversed(argspec.defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(argspec.args[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)

    return wrapper


class VecInt(torch.nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(torch.nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class LoadableModel(torch.nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape: Tuple[int, int, int],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()
        self.img_size = inshape
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        self.unet_model = PlainConvUNet(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=features_per_stage[0],
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=False,
            nonlin_first=False)

        # configure unet to flow field layer
        self.flow = conv_op(features_per_stage[0], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = torch.nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = torch.nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.integrate = VecInt(list(inshape), 5)

        # configure transformer
        self.stl = SpatialTransformer(inshape)
        self.stl_binary = SpatialTransformer(inshape, mode='nearest')

    def forward(self, input_channels):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            source_lbl: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # input_channels = [source, target] if 2 channels
        # input_channels = [source, target, source_lbl] if 3 channels
        x = torch.cat(input_channels, dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # integrate to produce diffeomorphic warp
        flow_field = self.integrate(flow_field)

        # warp image with flow field
        y_source = self.stl(input_channels[0], flow_field)
        return flow_field, y_source


class ConvBlock(torch.nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(torch.nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


def identity_grid(shape, unity=False):
    if unity:
        vectors = [torch.arange(0, s) / (s - 1) for s in shape]
    else:
        vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)
    return grid


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, shape, mode='bilinear', unity=False):
        super().__init__()
        self.shape = shape
        self.mode = mode
        self.unity = unity
        grid = identity_grid(shape=shape, unity=unity)
        self.register_buffer('grid', grid)
        # self.grid = identity_grid(shape=shape, unity=unity).to(device)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        if self.unity:
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] - 0.5)
        else:
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs.flip([-1])
        return torch.nn.functional.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode="border")


