"""
Jointly learn an invariant representation and a classifier on top of that
representation.
"""

import argparse
import lib
import torch

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='colored_mnist')
    parser.add_argument('--disc_dim', type=int, default=512)
    parser.add_argument('--prediction_method', type=str, default='random')
    parser.add_argument('--hparam_search', action='store_true')
    parser.add_argument('--l2reg_c', type=float, default=1e-3)
    parser.add_argument('--l2reg_d', type=float, default=1e-4)
    parser.add_argument('--lambda_erm', type=float, default=1.0)
    parser.add_argument('--lambda_gp', type=float, default=1.0)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--lr_d', type=float, default=1e-3)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--n_instances', type=int, default=16)
    parser.add_argument('--linear_classifier', action='store_true')
    parser.add_argument('--pca_dim', type=int, default=128)
    parser.add_argument('--steps', type=int, default=10001)
    parser.add_argument('--unwhitened', action='store_true')
    parser.add_argument('--z_dim', type=int, default=32)
    return parser.parse_args()

def main(args):
    print('Args:')
    for k,v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    default_hparams = {
        'l2reg_c': args.l2reg_c,
        'l2reg_d': args.l2reg_d,
        'lambda_erm': args.lambda_erm,
        'lambda_gp': args.lambda_gp,
        'lambda_orth': args.lambda_orth,
        'lr_d': args.lr_d,
        'lr_g': args.lr_g
    }

    def trial_fn(**hparams):
        dataset_fn = lib.datasets.REGISTRY[args.dataset]
        X_source, y_source, X_target, y_target = dataset_fn()
        whiten = (not args.unwhitened)
        source_pca = lib.pca.PCA(X_source, args.pca_dim, whiten=whiten)
        target_pca = lib.pca.PCA(X_target, args.pca_dim, whiten=whiten)
        X_source = source_pca.forward(X_source)
        X_target = target_pca.forward(X_target)

        # Apply random orthogonal transforms for optimization reasons.
        W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
        W2 = lib.ops.random_orthogonal_matrix(X_target.shape[1])
        X_source = X_source @ W1.T
        X_target = X_target @ W2.T

        source_rep, target_rep, classifier, divergences, target_accs = (
            lib.adversarial.train_dann(
                X_source, y_source, X_target, y_target, args.n_instances,
                batch_size=args.batch_size,
                disc_dim=args.disc_dim,
                linear_classifier=args.linear_classifier,
                steps=args.steps,
                z_dim=args.z_dim,
                **hparams))

        pred_fn = lib.dann_prediction_methods.REGISTRY[args.prediction_method]
        test_acc = pred_fn(
            classifier,
            divergences,
            target_rep,
            X_target,
            y_target,
            args.l2reg_c,
            args.linear_classifier,
            args.n_instances)

        # the 'top' prediction method is too high-variance to be used for
        # model selection, so we select based on mean accuracy instead.
        if args.prediction_method == 'top':
            val_acc = target_accs.mean()
        else:
            val_acc = test_acc

        return val_acc, test_acc

    if args.hparam_search:
        lib.hparam_search.hparam_search(trial_fn, default_hparams)
    else:
        val_acc, test_acc = trial_fn(**default_hparams)
        print('val_acc:', val_acc)
        print('test_acc:', test_acc)

if __name__ == '__main__':
    main(make_args())