"""
Jointly learn an invariant representation and a classifier on top of that
representation.
"""

import argparse
import lib
import torch

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='colored_mnist')
    parser.add_argument('--detach_Zs', action='store_true')
    parser.add_argument('--disc_dim', type=int, default=512)
    parser.add_argument('--hparam_search', action='store_true')
    parser.add_argument('--l2reg_c', type=float, default=0.)
    parser.add_argument('--l2reg_d', type=float, default=0.)
    parser.add_argument('--l2reg_r', type=float, default=0.)
    parser.add_argument('--lambda_erm', type=float, default=1.0)
    parser.add_argument('--lambda_gp', type=float, default=1.0)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--lr_d', type=float, default=1e-3)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--n_instances', type=int, default=16)
    parser.add_argument('--pca_dim', type=int, default=128)
    parser.add_argument('--rep_network', type=str, default='linear')
    parser.add_argument('--steps', type=int, default=100001)
    parser.add_argument('--unwhitened', action='store_true')
    parser.add_argument('--z_dim', type=int, default=32)
    return parser.parse_args()

def main(args):
    print('Args:')
    for k,v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    hparams = {
        'l2reg_c': args.l2reg_c,
        'l2reg_d': args.l2reg_d,
        'l2reg_r': args.l2reg_r,
        'lambda_erm': args.lambda_erm,
        'lambda_gp': args.lambda_gp,
        'lambda_orth': args.lambda_orth,
        'lr_d': args.lr_d,
        'lr_g': args.lr_g
    }

    dataset_fn = lib.datasets.REGISTRY[args.dataset]
    X_source, y_source, X_target, y_target = dataset_fn()
    X_source = lib.pca.PCA(X_source, args.pca_dim,
        whiten=(not args.unwhitened)).forward(X_source)
    X_target = lib.pca.PCA(X_target, args.pca_dim,
        whiten=(not args.unwhitened)).forward(X_target)
    X_source = X_source @ lib.ops.random_orthogonal_matrix(X_source.shape[1])
    X_target = X_target @ lib.ops.random_orthogonal_matrix(X_target.shape[1])

    if args.hparam_search:
        def trial_fn(**hparams):
            _, _, _, _, target_accs = lib.adversarial.train_dann(
                X_source, y_source, X_target, y_target, args.n_instances,
                batch_size=args.batch_size,
                detach_Zs=args.detach_Zs,
                disc_dim=args.disc_dim,
                rep_network=args.rep_network,
                steps=args.steps,
                z_dim=args.z_dim,
                **hparams
            )
            return target_accs.mean()
        hparams = lib.hparam_search.hparam_search(trial_fn, hparams)

    source_rep, target_rep, classifier, divergences, target_accs = \
        lib.adversarial.train_dann(
            X_source, y_source, X_target, y_target, args.n_instances,
            batch_size=args.batch_size,
            detach_Zs=args.detach_Zs,
            disc_dim=args.disc_dim,
            rep_network=args.rep_network,
            steps=args.steps,
            z_dim=args.z_dim,
            **hparams
    )

    results = [
        (name, fn(classifier, divergences, target_rep, X_target, y_target))
        for name, fn in lib.dann_prediction_methods.REGISTRY.items()
    ]

    print('Accuracy by prediction method:')
    for name, acc in results:
        print(name, acc.item())

if __name__ == '__main__':
    main(make_args())