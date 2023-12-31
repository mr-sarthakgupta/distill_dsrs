import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.fashionmnist_conv import Conv2FC2full, Conv2FC2simple
from archs.mnist_conv import Conv4FC3
from archs.wide_resnet_imagenet64 import wide_resnet_imagenet64
from datasets import get_normalize_layer, get_input_center_layer
from torch.nn.functional import interpolate
import torch.nn as nn
from archs.lenet import LeNet


# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", 'mnist_43', 'wide_resnet_imagenet64']

def get_architecture(arch: str, dataset: str, comment=None) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "fashion_22full":
        model = Conv2FC2full()
        model = model.cuda()
    elif arch == "fashion_22simple":
        model = Conv2FC2simple().cuda()
    elif arch == "mnist_43":
        model = Conv4FC3().cuda()
    elif arch == "lenet":
        model = LeNet(num_classes=10)
    elif arch == "wide_resnet_imagenet64":
        model = wide_resnet_imagenet64().cuda()
    # cohen uses normalize layer instead of input center layer
    if dataset == 'imagenet' and (comment is None or comment != 'orig-cohen'):
        normalize_layer = get_input_center_layer(dataset)
    else:
        normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
