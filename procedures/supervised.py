"""
Supervised source-domain training baseline.
"""

import argparse
import lib
import torch
import torch.nn.functional as F
from torch import nn, optim

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='colored_mnist')
    parser.add_argument('--hparam_search', action='store_true')
    parser.add_argument('--l2reg', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--linear_classifier', action='store_true')
    parser.add_argument('--pca_dim', type=int, default=128)
    parser.add_argument('--steps', type=int, default=10001)
    parser.add_argument('--unwhitened', action='store_true')
    return parser.parse_args()

def main(args):
    print('Args:')
    for k,v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    default_hparams = {
        'l2reg': args.l2reg,
        'lr': args.lr,
    }

    def trial_fn(**hparams):
        dataset_fn = lib.datasets.REGISTRY[args.dataset]
        X_source, y_source, X_target, y_target = dataset_fn()
        source_pca = lib.pca.PCA(X_source, args.pca_dim,
            whiten=(not args.unwhitened))
        X_source = source_pca.forward(X_source)
        X_target = source_pca.forward(X_target)

        # Apply random orthogonal transforms for optimization reasons.
        W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
        X_source = X_source @ W1.T
        X_target = X_target @ W1.T

        if args.linear_classifier:
            classifier = nn.Linear(args.pca_dim, int(y_source.max()+1)).cuda()
        else:
            classifier = nn.Sequential(
                nn.Linear(args.pca_dim, 128),
                nn.ReLU(),
                nn.Linear(128, int(y_source.max()+1))
            ).cuda()

        def forward():
            Xs, ys = lib.ops.get_batch([X_source, y_source], args.batch_size)
            return F.cross_entropy(classifier(Xs), ys)

        opt = optim.Adam(classifier.parameters(), lr=hparams['lr'],
            weight_decay=hparams['l2reg'])
        lib.utils.train_loop(forward, opt, args.steps)

        source_logits = classifier(X_source)
        target_logits = classifier(X_target)
        source_acc = lib.ops.multiclass_accuracy(source_logits, y_source).mean()
        target_acc = lib.ops.multiclass_accuracy(target_logits, y_target).mean()
        print('source_acc:', source_acc.item())
        return target_acc, target_acc

    if args.hparam_search:
        lib.hparam_search.hparam_search(trial_fn, default_hparams)
    else:
        val_acc, test_acc = trial_fn(**default_hparams)
        print('val_acc:', val_acc.item())
        print('test_acc:', test_acc.item())

if __name__ == '__main__':
    main(make_args())