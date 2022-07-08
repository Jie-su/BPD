from __future__ import print_function

import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append('/share/home/litaotao/JS/Disentangle_HAR_Server')
print(sys.path)

import argparse
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset_load import dataset_selection
from model.build_net import Disentangler, Generator, Classifier, Reconstructor, Mine
from tool.utils import _ent, _l2_rec

# Training settings
parser = argparse.ArgumentParser(description='PyTorch BPD Implementation')
parser.add_argument('--type_float', default=torch.FloatTensor)
parser.add_argument('--type_long', default=torch.LongTensor)

# Required training parameter
parser.add_argument('--max_epoch', type=int, metavar='N', help='maximum training epochs', required=True)
parser.add_argument('--dataset', type=str, metavar='N', help='selected dataset', required=True)
parser.add_argument('--input_dim', type=int, help='input dimension of backend net~(feature dim of dataset)',
                    required=True)
parser.add_argument('--output_dim', type=int, help='output dimension of backend net', required=True)
parser.add_argument('--cls_num', type=int, help='total activity class number', required=True)
parser.add_argument('--back_net', type=str, help='backend net', required=True)
parser.add_argument('--gpu', type=int, metavar='S', help='gpu device', required=True)
parser.add_argument('--lr', type=float, metavar='LR', help='learning rate (default: 0.0002)', required=True)
parser.add_argument('--target_domain', type=str, help='the target domain', required=True)

# Required dataset parameter
parser.add_argument('--win_len', type=int, default=30)
parser.add_argument('--n_domains', type=int, default=4, help='number of total domains actually')
parser.add_argument('--n_target_domains', type=int, default=1, help='number of target domains')
parser.add_argument('--position', type=str, default='wrist', help='position for the gotov dataset only')

# Default parameter for net
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed (default: 1)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N', help='source only or not')
parser.add_argument('--eval_only', type=int, default=0, help='evaluation only option')
parser.add_argument('--exp_name', type=str, default='cnn', metavar='N')
parser.add_argument('--use_cuda', action='store_true', default=True, help='Use cuda or not')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    args.type_float = torch.cuda.FloatTensor
    args.type_long = torch.cuda.LongTensor
print(args)

args.runing_directory = os.path.dirname(os.getcwd())
print(args.runing_directory)


def main():
    if args.back_net == 'cnn':
        args.exp_name = 'bpd_cnn'
    elif args.back_net == 'convlstmv2':
        args.exp_name = 'bpd_convlstmv2'

    if args.eval_only == 0:
        args.eval_only = False
    elif args.eval_only == 1:
        args.eval_only = True

    # loading dataset
    source_loaders, target_loader = dataset_selection(args)
    compose_dataset = [source_loaders, target_loader]

    # create solver object
    solver = Solver(args, batch_size=args.batch_size, candidate=args.target_domain,
                    dataset=args.dataset, win_len=args.win_len, learning_rate=args.lr,
                    optimizer=args.optimizer, checkpoint_dir=args.checkpoint_dir, data=compose_dataset)

    # start training
    for epoch in range(args.max_epoch):
        solver.train_epoch(epoch)
        if epoch % 1 == 0:
            solver.test(epoch)
        if epoch >= args.max_epoch:
            break


class Solver():
    def __init__(self, args, batch_size, candidate, dataset, win_len, learning_rate,
                 interval=1, optimizer='adam', checkpoint_dir=None, data=None):

        timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % args.exp_name + "_%s" % str(
            args.target_domain) + "_%s" % str(args.seed)

        self.dim = args.input_dim
        self.global_f1 = 0
        self.logdir = os.path.join('./logs', dataset, timestring)
        self.logger = SummaryWriter(log_dir=self.logdir)
        self.device = torch.device("cuda" if args.use_cuda else "cpu")
        self.result = []
        self.result_csv = self.logdir + str(args.max_epoch) + '_' + str(args.batch_size) \
                          + '_' + args.back_net + '_' + str(args.lr) + '.csv'

        self.class_num = args.cls_num
        self.back_net = args.back_net

        self.dataset = dataset
        self.candidate = candidate
        self.win_len = win_len
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.lr = learning_rate
        self.mi_coeff = 0.0001
        self.interval = interval
        self.mi_k = 1

        [self.source_loaders, self.target_loader] = data

        self.G = Generator(config=args)
        self.R = Reconstructor(config=args)
        self.MI = Mine(config=args)

        self.C = nn.ModuleDict({
            'ai': Classifier(config=args),
            'ni': Classifier(config=args),
        })

        self.D = nn.ModuleDict({
            'ai': Disentangler(config=args),
            'ni': Disentangler(config=args)
        })

        self.modules = nn.ModuleDict({
            'G': self.G,
            'R': self.R,
            'MI': self.MI
        })

        if args.eval_only:
            self.G.load_state_dict(torch.load(
                os.path.join(self.checkpoint_dir, str(self.dataset) + '_' + self.back_net + '_' + str(self.candidate),
                             str(self.dataset) + '-' + str(self.candidate) + '-bpd' + "-best-G.pt")))
            self.D.load_state_dict(torch.load(
                os.path.join(self.checkpoint_dir, str(self.dataset) + '_' + self.back_net + '_' + str(self.candidate),
                             str(self.dataset) + '-' + str(self.candidate) + '-bpd' + "-best-D.pt")))
            self.C.load_state_dict(torch.load(
                os.path.join(self.checkpoint_dir, str(self.dataset) + '_' + self.back_net + '_' + str(self.candidate),
                             str(self.dataset) + '-' + str(self.candidate) + '-bpd' + "-best-C.pt")))
        self.xent_loss = nn.CrossEntropyLoss().cuda()
        self.adv_loss = nn.BCEWithLogitsLoss().cuda()
        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.to_device()

    def to_device(self):
        for k, v in self.modules.items():
            self.modules[k] = v.cuda()

        for k, v in self.C.items():
            self.C[k] = v.cuda()

        for k, v in self.D.items():
            self.D[k] = v.cuda()

    def set_optimizer(self, which_opt='adam', lr=0.001, momentum=0.9):
        self.opt = {
            'C_ai': optim.Adam(self.C['ai'].parameters(), lr=lr, weight_decay=5e-4),
            'C_ni': optim.Adam(self.C['ni'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ai': optim.Adam(self.D['ai'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ni': optim.Adam(self.D['ni'].parameters(), lr=lr, weight_decay=5e-4),
            'G': optim.Adam(self.G.parameters(), lr=lr, weight_decay=5e-4),
            'R': optim.Adam(self.R.parameters(), lr=lr, weight_decay=5e-4),
            'MI': optim.Adam(self.MI.parameters(), lr=lr, weight_decay=5e-4),
        }

    def reset_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()

    def mi_estimator(self, x, y, y_):
        joint, marginal = self.MI(x, y), self.MI(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def group_opt_step(self, opt_keys):
        for k in opt_keys:
            self.opt[k].step()
        self.reset_grad()

    def optimize_classifier(self, data, label, epoch):
        feat = self.G(data)
        _loss = dict()
        for key in ['ai', 'ni']:
            _loss['class_' + key] = self.xent_loss(
                self.C[key](self.D[key](feat)), label)

        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['G', 'C_ai', 'C_ni', 'D_ai', 'D_ni'])
        return _loss

    def mutual_information_minimizer(self, feature):
        for i in range(0, self.mi_k):
            activity, noise = self.D['ai'](self.G(feature)), self.D['ni'](self.G(feature))
            noise_shuffle = torch.index_select(
                noise, 0, torch.randperm(noise.shape[0]).to(self.device))
            MI_act_noise = self.mi_estimator(activity, noise, noise_shuffle)
            MI = 0.25 * (MI_act_noise) * self.mi_coeff
            MI.backward()
            self.group_opt_step(['D_ai', 'D_ni', 'MI'])

    def class_confusion(self, data):
        # - adversarial training
        # f_ci = CI(G(im)) extracts features that are class irrelevant
        # by maximizing the entropy, given that the classifier is fixed
        loss = _ent(self.C['ni'](self.D['ni'](self.G(data))))
        loss.backward()
        self.group_opt_step(['D_ni', 'G'])
        return loss

    def optimize_rec(self, data):
        feature = self.G(data)
        feat_ai, feat_ni = self.D['ai'](feature), self.D['ni'](feature)
        rec_feat = self.R(torch.cat([feat_ai, feat_ni], 1))
        recon_loss = _l2_rec(rec_feat, feature)
        recon_loss.backward()
        self.group_opt_step(['D_ai', 'D_ni', 'R'])
        return recon_loss

    def train_epoch(self, epoch):
        # set training
        for k in self.modules.keys():
            self.modules[k].train()
        for k in self.C.keys():
            self.C[k].train()
        for k in self.D.keys():
            self.D[k].train()

        # Load training set and testing set with LOSO setting
        pbar_descr_prefix = "Epoch %d" % (epoch)
        with tqdm(total=10000, ncols=80, dynamic_ncols=False,
                  desc=pbar_descr_prefix) as pbar:
            for source_loader in self.source_loaders:
                for batch_idx, (data, label, domain) in enumerate(source_loader):
                    data = torch.FloatTensor(data.float()).permute(0, 2, 1).to(self.device)
                    label = label.long().to(self.device)
                    self.reset_grad()
                    # ================================== #
                    class_loss = self.optimize_classifier(data, label, epoch)
                    self.mutual_information_minimizer(data)
                    confusion_loss = self.class_confusion(data)
                    recon_loss = self.optimize_rec(data)
                    self.logger.add_scalar("confusion_loss", confusion_loss.detach().cpu().numpy(),
                                           global_step=batch_idx)
                    self.logger.add_scalar("rec_loss", recon_loss.detach().cpu().numpy(), global_step=batch_idx)
                    # ================================== #
                    if (batch_idx + 1) % self.interval == 0:
                        # ================================== #
                        for key, val in class_loss.items():
                            self.logger.add_scalar(
                                "class_loss/%s" % key, val,
                                global_step=batch_idx)

                    pbar.update()

    def test(self, epoch):
        """
        compute the accuracy over the supervised training set or the testing set
        """
        # set evaluation modal for Generator, Disentangler, and Classifer
        self.G.eval()
        self.D['ai'].eval()
        self.C['ai'].eval()

        size = 0
        correct_aa, correct_an, correct_nn = 0, 0, 0
        y_true = []
        y_pre_aa = []

        with torch.no_grad():
            for batch_idx, (data, label, _) in enumerate(self.target_loader):
                # Get loaded dataset
                data, label = data.float().permute(0, 2, 1).to(self.device), label.long().to(self.device)
                # ouput extracted feature from generator
                feat = self.G(data)
                # prediction result from C['ai']->D['ai']
                pre_aa = self.C['ai'](self.D['ai'](feat))
                # append result to the corresponding list
                y_pre_aa.append(pre_aa.view(-1, self.class_num))
                y_true.append(label.view(-1))
                # calculate correctness of predict label and count number
                correct_aa += pre_aa.data.max(1)[1].eq(label.data).cpu().sum()
                size += label.data.size()[0]

        # concatenate all prediction
        y_pre_aa = torch.cat(y_pre_aa, 0)
        y_true = torch.cat(y_true, 0)
        # calculate f1 socre for the activity recognition on target domain
        f1 = f1_score(y_true.cpu(), y_pre_aa.cpu().max(dim=-1)[1], average='macro')
        # calculate acc for aa/an and nn
        acc_aa = 100. * correct_aa / size
        # If the f1 socre is higher than the global f1 score, then store it
        f1_sc = f1
        if f1_sc > self.global_f1:
            self.global_f1 = f1_sc
            torch.save(self.G.state_dict(),
                       os.path.join(self.logdir,
                                    str(self.dataset) + '-' + str(self.candidate) + "-bpd-best-G.pt"))
            torch.save(self.D.state_dict(),
                       os.path.join(self.logdir,
                                    str(self.dataset) + '-' + str(self.candidate) + "-bpd-best-D.pt"))
            torch.save(self.C.state_dict(),
                       os.path.join(self.logdir,
                                    str(self.dataset) + '-' + str(self.candidate) + "-bpd-best-C.pt"))

        print('\nTest set||aa_acc: {:.2f}%||aa_F1 score: {:.2f}% \n'.format(acc_aa, f1_sc * 100))

        self.result.append([acc_aa, f1_sc, self.global_f1, epoch])
        result_np = np.array(self.result, dtype=float)
        np.savetxt(self.result_csv, result_np, fmt='%.4f', delimiter=',')

        # add scalar to tensorboard
        self.logger.add_scalar(
            "test_target_acc/acc", acc_aa,
            global_step=epoch)

        self.logger.add_scalar(
            "test_target_acc/F1_score", f1_sc,
            global_step=epoch)


if __name__ == '__main__':
    main()
