from zennit.rules import Epsilon, AlphaBeta, ZPlus, Norm, Pass, Gamma, ZBox
from zennit.types import Linear, Convolution, Activation, AvgPool
from zennit.layer import Sum
import torch.nn as nn
import torch
from zennit.core import Hook


class Ignore(Hook):
    '''Ignore the module's gradient and pass through the output gradient.'''

    def backward(self, module, grad_input, grad_output):
        '''Directly return grad_output.'''
        return tuple([torch.zeros_like(elem) for elem in grad_input])


def module_map_resnet(ctx, name, module):
    # check whether there is at least one child, i.e. the module is not a leaf
    try:
        next(module.children())
    except StopIteration:
        # StopIteration is raised if the iterator has no more elements,
        # which means in this case there are no children and module is a leaf
        pass
    else:
        # if StopIteration is not raised on the first element, module is not a leaf
        return None

    # if the module is not Linear, we do not want to assign a hook

    if name == 'conv1':
        return ZBox(low=-3, high=3)

    if isinstance(module, Convolution):
        if module.stride == 1 or all(elem == 1 for elem in module.stride) or not 'downsample' in name:
            return Gamma(gamma=0.25)

        return Ignore()

    if isinstance(module, nn.Linear):
        return Epsilon(epsilon=1e-3)

    if isinstance(module, (Sum, AvgPool)):
        return Norm()

    if isinstance(module, Activation):
        return Pass()

    # all other rules should be assigned Epsilon
    return None
