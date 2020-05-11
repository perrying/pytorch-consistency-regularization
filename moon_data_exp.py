"""
Two moons experiment for visualization
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tqdm import tqdm

from ssl_lib.algs.builder import gen_ssl_alg
from ssl_lib.models.utils import ema_update
from ssl_lib.consistency.builder import gen_consistency


def gen_model():
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )


def gen_ssl_moon_dataset(seed, num_samples, labeled_sample, noise_factor=0.1):
    assert num_samples > labeled_sample
    data, label = make_moons(num_samples, False, noise_factor, random_state=seed)
    data = (data - data.mean(0, keepdims=True)) / data.std(0, keepdims=True)

    l0_idx = (label == 0)
    l1_idx = (label == 1)

    l0_data = data[l0_idx]
    l1_data = data[l1_idx]

    np.random.seed(seed)

    l0_data = np.random.permutation(l0_data)
    l1_data = np.random.permutation(l1_data)

    labeled_l0 = l0_data[:labeled_sample//2]
    labeled_l1 = l1_data[:labeled_sample//2]

    unlabeled = np.concatenate([
        l0_data[labeled_sample//2:], l1_data[labeled_sample//2:]
    ])

    l0_label = np.zeros(labeled_l0.shape[0])
    l1_label = np.ones(labeled_l1.shape[0])
    label = np.concatenate([l0_label, l1_label])

    return labeled_l0, labeled_l1, unlabeled, label


def scatter_plot_with_confidence(l0_data, l1_data, all_data, model, device, out_dir=None, show=False):
    xx, yy = np.meshgrid(
        np.linspace(all_data[:,0].min()-0.1, all_data[:,0].max()+0.1, 1000),
        np.linspace(all_data[:,1].min()-0.1, all_data[:,1].max()+0.1, 1000))
    np_points = np.stack([xx.ravel(),yy.ravel()],1).reshape(-1, 2)
    points = torch.from_numpy(np_points).to(device).float()
    outputs = model(points).softmax(1)[:,1].detach().to("cpu").numpy().reshape(xx.shape)
    plt.contourf(xx, yy, outputs, alpha=0.5, cmap=plt.cm.jet)
    plt.scatter(all_data[:,0], all_data[:,1], c="gray")
    plt.scatter(l0_data[:,0], l0_data[:,1], c="blue")
    plt.scatter(l1_data[:,0], l1_data[:,1], c="red")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.grid()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, "confidence_with_labeled.png"))
    if show:
        plt.show()
    plt.contourf(xx, yy, outputs, alpha=0.5, cmap=plt.cm.jet)
    plt.scatter(l0_data[:,0], l0_data[:,1], c="blue")
    plt.scatter(l1_data[:,0], l1_data[:,1], c="red")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.grid()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, "confidence.png"))
    if show:
        plt.show()


def scatter_plot(l0_data, l1_data, unlabeled_data, out_dir=None, show=False):
    plt.scatter(unlabeled_data[:,0], unlabeled_data[:,1], c="gray")
    plt.scatter(l0_data[:,0], l0_data[:,1], c="blue")
    plt.scatter(l1_data[:,0], l1_data[:,1], c="red")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.grid()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, "labeled_raw_data.png"))
    if show:
        plt.show()
    plt.scatter(l0_data[:,0], l0_data[:,1], c="blue")
    plt.scatter(l1_data[:,0], l1_data[:,1], c="red")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.grid()
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, "raw_data.png"))
    if show:
        plt.show()

def fit(cfg):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"

    model = gen_model().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), cfg.lr)

    weak_augmentation = lambda x: x + torch.randn_like(x) * cfg.gauss_std

    # set consistency type
    consistency = gen_consistency(cfg.consistency, cfg)
    # set ssl algorithm
    ssl_alg = gen_ssl_alg(
        cfg.alg,
        cfg
    )

    l0_data, l1_data, u_data, label = gen_ssl_moon_dataset(
        cfg.seed, cfg.n_sample, cfg.n_labeled, cfg.noise_factor
    )

    labeled_data = np.concatenate([l0_data, l1_data])

    scatter_plot(l0_data, l1_data, u_data, cfg.out_dir, cfg.vis_data)

    tch_labeled_data = torch.from_numpy(labeled_data).float().to(device)
    tch_unlabeled_data = torch.from_numpy(u_data).float().to(device)
    label = torch.from_numpy(label).long().to(device)

    for i in range(cfg.iterations):
        unlabeled_weak1 = weak_augmentation(tch_unlabeled_data)
        unlabeled_weak2 = weak_augmentation(tch_unlabeled_data)
        all_data = torch.cat([
            tch_labeled_data,
            unlabeled_weak1,
            unlabeled_weak2], 0)

        outputs = model(all_data)
        labeled_logits = outputs[:tch_labeled_data.shape[0]]
        loss = F.cross_entropy(labeled_logits, label)
        if cfg.coef > 0:
            unlabeled_logits, unlabeled_logits_target = torch.chunk(outputs[tch_labeled_data.shape[0]:], 2, dim=2)

            y, targets, mask = ssl_alg(
                stu_preds = unlabeled_logits,
                tea_logits = unlabeled_logits_target.detach(),
                w_data = unlabeled_weak1,
                s_data = unlabeled_weak2,
                stu_forward = model,
                tea_forward = model
            )

            L_consistency = consistency(y, targets, mask)
            loss += cfg.coef * L_consistency
        else:
            L_consistency = torch.zeros_like(loss)

        if cfg.entropy_minimize > 0:
            loss -= cfg.entropy_minimize * (unlabeled_logits.softmax(1) * F.log_softmax(unlabeled_logits, 1)).sum(1).mean()

        print("[{}/{}] loss {} | ssl loss {}".format(
            i+1, cfg.iterations, loss.item(), L_consistency.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scatter_plot_with_confidence(l0_data, l1_data, all_data, model, device, cfg.out_dir, cfg.vis_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--n_sample", default=1000, type=int, help="number of samples")
    parser.add_argument("--n_labeled", default=10, type=int, help="number of labeled samples")
    parser.add_argument("--noise_factor", default=0.1, type=float, help="std of gaussian noise")
    # optimization config
    parser.add_argument("--iterations", default=1000, type=int, help="number of training iteration")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    # SSL common config
    parser.add_argument("--alg", default="cr", type=str, help="ssl algorithm, ['ict', 'cr', 'pl', 'vat']")
    parser.add_argument("--coef", default=1, type=float, help="coefficient for consistency loss")
    parser.add_argument("--ema_teacher", action="store_true", help="consistency with mean teacher")
    parser.add_argument("--ema_factor", default=0.999, type=float, help="exponential mean avarage factor")
    parser.add_argument("--entropy_minimize", "-em", default=0, type=float, help="coefficient of entropy minimization")
    parser.add_argument("--threshold", default=None, type=float, help="pseudo label threshold")
    parser.add_argument("--sharpen", default=None, type=float, help="tempereture parameter for sharpening")
    parser.add_argument("--temp_softmax", default=None, type=float, help="tempereture for softmax")
    parser.add_argument("--gauss_std", default=0.1, type=float, help="standard deviation for gaussian noise")
    ## SSL alg parameter
    ### ICT config
    parser.add_argument("--alpha", default=0.1, type=float, help="parameter for beta distribution in ICT")
    ### VAT config
    parser.add_argument("--eps", default=6, type=float, help="norm of virtual adversarial noise")
    parser.add_argument("--xi", default=1e-6, type=float, help="perturbation for finite difference method")
    parser.add_argument("--vat_iter", default=1, type=int, help="number of iteration for power iteration")
    ## consistency config
    parser.add_argument("--consistency", "-consis", default="ce", type=str, help="consistency type, ['ce', 'ms']")
    parser.add_argument("--sinkhorn_tau", default=10, type=float, help="tempereture parameter for sinkhorn distance")
    parser.add_argument("--sinkhorn_iter", default=10, type=int, help="number of iterations for sinkhorn normalization")
    # evaluation config
    parser.add_argument("--weight_average", action="store_true", help="evaluation with weight-averaged model")
    # misc
    parser.add_argument("--out_dir", default="log", type=str, help="output directory")
    parser.add_argument("--seed", default=96, type=int, help="random seed")
    parser.add_argument("--vis_data", action="store_true", help="visualize input data")

    args = parser.parse_args()

    fit(args)
