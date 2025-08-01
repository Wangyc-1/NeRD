import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from time import time
import statistics as st
from tqdm import tqdm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.distributed as dist
from neighbor_filter import adjust_neighbors
from kmeans_pseudo_update import run_kmeans_update
from compute_total_loss import compute_total_loss
torch.autograd.set_detect_anomaly(True)

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()


    txt_src = [folder + args.dset + '/' + s for s in txt_src]
    txt_tar = [folder + args.dset + '/' + s for s in txt_tar]
    txt_test = [folder + args.dset + '/' + s for s in txt_test]

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(loader, netF, netB, netC):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            outputs = netC(fea)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    harmonic = st.harmonic_mean(acc)
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = " ".join(aa)
    return aacc, acc, harmonic

def hyper_decay(x, beta=-2, alpha=1):
    weight = (1 + 10 * x) ** (-beta) * alpha
    return weight


def extract_features(loader, netF, netB=None):
    netF.eval()
    if netB:
        netB.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in loader:
            inputs, lbls, _ = data
            inputs = inputs.cuda()
            feat = netF(inputs)
            if netB:
                feat = netB(feat)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels

def plot_tsne(features_before, labels,title_before="t-SNE Before Training"):
    tsne = TSNE(n_components=2, random_state=42)
    emb_before = tsne.fit_transform(features_before)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=emb_before[:, 0], y=emb_before[:, 1], hue=labels, palette="deep", s=5,legend=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("The t-SNE visualization of the optimized feature space")
    legend = plt.legend(title="Classes", loc="lower right", bbox_to_anchor=(1, 0),
                        markerscale=0.6, fontsize=5, frameon=True)
    plt.savefig(args.output_dir+"optimized", dpi=300, bbox_inches='tight')
    plt.close()

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    modelpath = args.output_dir_src + 'base_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + 'base_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + 'base_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netF_ = network.ResBase(res_name=args.net).cuda()
    netB_ = network.feat_bootleneck(
        type=args.classifier,
        feature_dim=netF.in_features,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    netC_ = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()
    for param_q, param_k in zip(netF.parameters(), netF_.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient
    for param_q, param_k in zip(netB.parameters(), netB_.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient
    for param_q, param_k in zip(netC.parameters(), netC_.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
        param_k.requires_grad = False  # not update by gradient

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        param_group += [{"params": v, "lr": args.lr * args.lr_F}]  # 0.1
    for k, v in netB.named_parameters():
        param_group += [{"params": v, "lr": args.lr * args.lr_B}]  # 1
    for k, v in netC.named_parameters():
        param_group_c += [{"params": v, "lr": args.lr * args.lr_C}]  # 1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, 12)
    label_bank = torch.randn(num_sample)
    pseudo_bank = torch.randn(num_sample).long()

    netF.eval()
    netB.eval()
    netC.eval()

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels, indx = next(iter_test)
            inputs = inputs.cuda()
            labels = labels.type(torch.FloatTensor)
            output = netB(netF(inputs))
            output_norm = F.normalize(output)
            outputs_ = netC(output)
            outputs = nn.Softmax(-1)(outputs_)
            pseudo_label = torch.argmax(outputs, 1)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone().cpu()
            label_bank[indx] = labels
            pseudo_bank[indx] = pseudo_label.detach().clone().cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])

    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    real_max_iter = max_iter
    epoch = 0
    with tqdm(total=real_max_iter) as pbar:
        while iter_num < real_max_iter:
            start = time()
            try:
                inputs_test, _, tar_idx = next(iter_test)
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_test)

            if inputs_test.size(0) == 1:
                continue

            inputs_test = inputs_test.cuda()
            mu = torch.mean(inputs_test, dim=0)
            sigma = torch.sqrt(torch.mean((inputs_test - mu) ** 2, dim=0))
            epsilon = torch.randn_like(inputs_test) * sigma
            inputs_test_perturbed = inputs_test + epsilon
            if True:
                alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
            else:
                alpha = args.alpha

            iter_num += 1
            pbar.update(1)
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

            features_test = netB(netF(inputs_test))
            output_f_norm = F.normalize(features_test)
            w = F.normalize(netC.fc.weight_v) * netC.fc.weight_g
            # print("fea{}".format(features_test.shape))
            pred_test = netC(features_test)
            score_test = nn.Softmax(dim=1)(pred_test)
            # print("sco{}".format(score_test.shape)) (64,12)
            pseudo_label = torch.argmax(score_test, 1).detach()
            top2 = torch.topk(score_test, 2).values
            margin = top2[:,0] - top2[:,1]

            with torch.no_grad():
                output_f_ = output_f_norm.cpu().detach().clone()
                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = score_test.detach().clone().cpu()
                pseudo_bank[tar_idx] = pseudo_label.detach().clone().cpu()
                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]
                _, idx_far = torch.topk(distance, dim=-1, largest=False, k=args.K+1)
                idx_far = idx_far[:, 1:]  # batch x K
                score_far = score_bank[idx_far]
                adjusted_idx_near, score_nn = adjust_neighbors(
                    idx_near, output_f_, fea_bank, pseudo_bank, score_bank, args
                )
                pseudo_bank = run_kmeans_update(iter_num, fea_bank, pseudo_bank, score_bank)
                distance = output_f_ @ fea_bank.T
                _, idx_aug_target = torch.topk(distance, dim=-1, largest=True, k=64)
                distance_sim = output_f_ @ fea_bank.T
                _, idx_aug_source = torch.topk(distance_sim, dim=-1, largest=True, k=64)
                num_augmentations = 1
                perturbations = []
                for _ in range(num_augmentations):
                    neighbor_feat_target = fea_bank[idx_aug_target]
                    perturb_mean = torch.mean(neighbor_feat_target, dim=1).cuda()
                    neighbor_feat_source = fea_bank[idx_aug_source]
                    perturb_std = torch.std(neighbor_feat_source, dim=1).cuda()
                    z_aug = perturb_mean + perturb_std * torch.randn_like(perturb_std)
                    perturbed_logits = netC(z_aug)
                    perturbed_probs = nn.Softmax(dim=1)(perturbed_logits)
                    perturbations.append(perturbed_probs)
            loss = compute_total_loss(inputs_test, score_test, score_near, score_far,
                                      perturbations, features_test, pseudo_label, args, alpha)
            optimizer.zero_grad()
            optimizer_c.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_c.step()
            pbar.set_description(f'iter_num:{iter_num}; time:{time()-start:.2f} sec; ')
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                epoch += 1
                start_inference = time()
                netF.eval()
                netB.eval()
                netC.eval()
                if args.dset == "visda-2017":
                    acc, accc, harmonic = cal_acc(
                        dset_loaders["test"],
                        netF,
                        netB,
                        netC,
                    )
                    log_str = (
                        "Task: {}, Iter:{}/{}; epoch:{}; Arithmetic: {:.2f}".format(
                            args.name, iter_num, max_iter, epoch, acc
                        )
                        + "\n"
                        + "T: "
                        + accc
                    )

                args.out_file.write(log_str + "\n")
                args.out_file.flush()
                print("\n" + log_str + "\n")
                netF.train()
                netB.train()
                netC.train()
                pbar.set_description(f'inference time:{time()-start_inference:.2f} sec')
    features_before, labels = extract_features(dset_loaders["target"], netF)
    print("Extracted initial features for t-SNE visualization.")
    if args.issave:
        args.output_dir += f'/{str(args.seed)}/'
        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F.pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B.pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C.pt"))


    plot_tsne(features_before, labels)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="NeRD")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument("--s", type=int, default=0, help="soure")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=8, help="number of workers")
    parser.add_argument("--dset", type=str, default="visda-2017")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate") #1e-3
    parser.add_argument("--net", type=str, default="resnet101")
    parser.add_argument("--seed", type=int, default=2019, help="random seed") #2021
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha_1", type=float, default=0.1)
    parser.add_argument("--alpha_2", type=float, default=0.2)
    parser.add_argument("--pert", type=float, default=-0.1)
    parser.add_argument("--nn", type=float, default=-0.1)
    parser.add_argument("--far", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--lambda_0", type=float, default=5.0)
    parser.add_argument("--lr_F", type=float, default=0.1)
    parser.add_argument("--aa", type=float, default=0.01)
    parser.add_argument("--bb", type=float, default=0.01)
    parser.add_argument("--dd", type=float, default=0.01)
    parser.add_argument("--lr_B", type=float, default=1.0)
    parser.add_argument("--lr_C", type=float, default=1.0)
    parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=1.0)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--output", type=str, default=" ")
    parser.add_argument("--output_src", type=str, default=" ")
    parser.add_argument("--da", type=str, default="uda")
    parser.add_argument("--issave", type=bool, default=False)
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha_decay", default=True)
    parser.add_argument("--nuclear", default=False, action="store_true")
    parser.add_argument("--var", default=False, action="store_true")


    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num = 65
    if args.dset == "visda-2017":
        names = ["train", "validation"]
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = ''
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.output_dir_src = osp.join(
            args.output_src,
        )
        args.output_dir = osp.join(
            args.output,
        )
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(
            osp.join(args.output_dir, "NeRD.txt".format(args.seed)), "w"  )
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()

        train_target(args)
