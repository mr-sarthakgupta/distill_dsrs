from multiprocessing.pool import Pool, ThreadPool
import argparse
from tensorboardX import SummaryWriter, GlobalSummaryWriter
import os
from time import time, sleep
from random import random
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from icecream import ic

from scipy.stats import gamma

from datasets import get_input_dimension
from algo.algo import calc_fast_beta_th, check
from utils import read_pAs, read_orig_Rs
from distribution import StandardGaussian, GeneralGaussian, LinftyGaussian, LinftyGeneralGaussian, L1GeneralGaussian
from train import simple_train, test
from train_utils import prologue
from architectures import ARCHITECTURES
from datasets import DATASETS

# ====== global settings =======
workers = 10
# workers can be changed by argparse
ORIG_R_EPS = 5e-5
DUAL_EPS = 1e-8
# The above eps works well (and guarantees soundness in practical data) if using argparse's precision
# =============================




dist = None

def orig_radius_pool_func(args):
    """
        Paralleled original radius computing function
    :param args:
    :return:
    """
    no, pA, _ = args
    stime = time()
    print('On #', no)
    r = dist.certify_radius(pA)
    print('#', no, 'done:', time() - stime, 's')
    return r, time() - stime


def distill(base_dir, out_base_dir, disttype, d, k, std, aux_stds, N, alpha):
    """
        The entrance function, or the dispatcher, for the original radius computation
    :param base_dir:
    :param out_base_dir:
    :param disttype:
    :param d:
    :param k:
    :param std:
    :param aux_stds:
    :param N:
    :param alpha:
    :return: distilled model
    """
    smooth_outs = torch.load(os.path.join(out_base_dir, f'smooth_outs-{disttype}-{k}-{std}-{N}-{alpha}.pth'))
    initial_train_loader, test_loader, criterion, model, optimizer, scheduler, \
    starting_epoch, logfilename, model_path, device, writer = prologue(args)
    pin_memory = (args.dataset == "imagenet")
    smooth_out_loader = DataLoader(smooth_outs, shuffle=True, batch_size=args.batch, num_workers=args.workers, pin_memory=pin_memory)
    smooth_data_list = []
    for i, ((x, y), smooth_preds) in enumerate(zip(initial_train_loader, smooth_out_loader)):
        smooth_data_list.append((x, smooth_preds))
        
    smooth_dataloader = DataLoader(smooth_data_list, shuffle=True, batch_size=None, num_workers=args.workers, pin_memory=pin_memory)

    step_counter = {'step': 0}
    for epoch in range(starting_epoch, args.epochs):
        if args.dataset != 'imagenet':
            if args.k == 0:
                now_k = 0
            else:
                now_k = math.ceil(args.k - args.k * math.exp(- epoch * math.log(args.k) / args.k_warmup)) \
                    if epoch <= args.k_warmup else args.k
            print(f'Epoch {epoch} with k = {now_k}')

        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False).cuda()
        if args.dataset != 'imagenet':
            model = simple_train(loader = smooth_dataloader, model = model, criterion = criterion, optimizer = optimizer, epoch = epoch, k = now_k, device = device, writer = writer, step_counter = step_counter, k_warmup = args.k_warmup)
        else:
            model = simple_train(smooth_dataloader, model, criterion, optimizer, epoch, args.k, args.mix_infty,
                                          args.noise_sd, device, writer, args.k_warmup, step_counter)
        if args.dataset != 'imagenet':
            model = test(test_loader, model, criterion, epoch, now_k,
                                       args.noise_sd, device, writer, args.print_freq)
        else:
            if args.k == 0:
                now_k = 0
            else:
                now_k = math.ceil(args.k - args.k * math.exp(- step_counter['step'] * math.log(args.k) / args.k_warmup))\
                    if step_counter['step'] <= args.k_warmup else args.k
            test_loss, test_acc = test(test_loader, model, criterion, epoch, now_k,
                                       args.noise_sd, device, writer, args.print_freq)
        
        # log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
        #     epoch, after - before,
        #     scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step(epoch)
    return model


parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--model', default='consistency-cifar-0.50.pth', type=str, help='model name')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=50,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# somehow doesn't work...
# parser.add_argument('--gpu', default=None, type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')

#####################
# Options added by Salman et al. (2019)
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')

#####################
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=20., type=float)

# Options when SmoothAdv is used (Salman et al., 2019)
parser.add_argument('--adv-training', action='store_true')
parser.add_argument('--epsilon', default=512, type=float)
parser.add_argument('--num-steps', default=4, type=int)
parser.add_argument('--warmup', default=10, type=int, help="Number of epochs over which "
                                                           "the maximum allowed perturbation increases linearly "
                                                           "from zero to args.epsilon.")

parser.add_argument('--k', default=1530, type=int, help="Final general Gaussian parameter")
parser.add_argument('--k-warmup', default=100, type=int, help="Number of epochs over which the general Gaussian "
                                                              "parameter increases from zero to desired k")
parser.add_argument('--infty', default=0, type=int, help="whether to use pure infty radial distribution")
parser.add_argument('--mix-infty', default=0, type=int, help="How many batches mix the infty norm noises")
parser.add_argument('--mix-infty-multipler', default=1, type=float, help="The variance ratio between infty and l2")
parser.add_argument('--l1', default=0, type=int, help="whether to use pure L1 radial distribution")

# parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'imagenet', 'tinyimagenet'],
#                     help='We rely on the dataset information to obtain the input dimension.'
#                                 'New datasets can be easily integrated.')
parser.add_argument('--disttype', default='general-gaussian', choices=['general-gaussian', 'gaussian',
                                         'general-gaussian-th', 'gaussian-th'],
                    help='general-gaussian: generalized Gaussian as P, generalized Gaussian with a different variance as Q;'
                                'gaussian: standard Gaussian as P, standard Gaussian with a different variance as Q (this option obtains invisible improvements when input dimension >= 40);'
                                'general-gaussian-th: generalized Gaussian as P, generalzied Gaussian with threshold cutting as Q;'
                                'gaussian-th: standard Gaussian as P, generalzied Gaussian with threshold cutting as Q (this option obtains invisible improvements when input dimension >= 40).')
parser.add_argument('--sigma', default=0.5, type=float,
                    help="sigma of P distribution. Note: as recorded in the paper, here it is sigma instead of sigma'")
parser.add_argument('--N', default=10, type=int,
                    help='Sampling number. This parameter does not make a difference to the computational method itself. '
                                'It is just for extracting the correct sampling info file whose name includes the sampling number information.')
parser.add_argument('--alpha', default=0.0005, type=float,
                    help='The certification confidence. Similar to N, this parameter does not make a difference to the computational method itself')
parser.add_argument('-b', action='append', nargs='+',
                    help='The sigma of Q distribution (if using general-gaussian or gaussian as disttype), or '
                                'the percentile of thresholding if a real number between 0 and 1 or percentitle selection heuristics if x, x2, x+, or x2+ (if using general-gaussian-th or gaussian-th as disttype)'
                                '+ insteads for heuristic that includes the fall-back strategy. x+ has the best performance empirically.'
                                'Multiple options can be specified at the same time, and the script will run the certification for them respectively.')

parser.add_argument('--sampling_dir', type=str, default='data/sampling',
                    help="folder for extracting the sampling pA info.")
parser.add_argument('--original_rad_dir', type=str, default='data/sampling',
                    help="folder for extracting (if task = improved) or outputing (if task = origin) the original radius computed by Neyman-Pearson")

args = parser.parse_args()

args.outdir = "placeholder"

if __name__ == '__main__':
    workers = args.workers

    d = get_input_dimension(args.dataset)

    # init for fast Gaussian computation
    calc_fast_beta_th(d)

    k = args.k
    sigma = args.sigma
    if args.b is not None:
        betas = args.b[0]
    else:
        betas = list()
    N = args.N
    alpha = args.alpha

    betas = [float(b) if isinstance(b, str) and b[0].isdigit() else b for b in betas]

    print(f"""
==============
Metainfo:
    model = {args.model}
    distype = {args.disttype}
    d = {d}
    k = {k}
    sigma = {sigma}
    betas = {betas}
    N = {N}
    alpha = {alpha}
==============
""")

    distilled_model = distill(os.path.join(args.sampling_dir, args.model), os.path.join(args.original_rad_dir, args.model), args.disttype, d, k, sigma, betas, N, alpha)
    torch.save(distilled_model, f"distilled_smooth_{args.model}")