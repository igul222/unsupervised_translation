"""
Find a linear source-to-target translation.
"""

import argparse
import lib

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='colored_mnist')
    parser.add_argument('--disc_dim', type=int, default=512)
    parser.add_argument('--lambda_gp', type=float, default=1.0)
    parser.add_argument('--lambda_orth', type=float, default=0.1)
    parser.add_argument('--lr_d', type=float, default=1e-3)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--n_instances', type=int, default=8)
    parser.add_argument('--pca_dim', type=int, default=128)
    parser.add_argument('--steps', type=int, default=10001)
    parser.add_argument('--l2reg_d', type=float, default=1e-4)
    return parser.parse_args()

def main(args):
    print('Args:')
    for k,v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    X_source, _, X_target, _ = lib.datasets.REGISTRY[args.dataset]()

    source_pca = lib.pca.PCA(X_source, args.pca_dim, whiten=True)
    target_pca = lib.pca.PCA(X_target, args.pca_dim, whiten=True)
    X_source = source_pca.forward(X_source)
    X_target = target_pca.forward(X_target)

    # Apply random orthogonal transforms.
    W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
    W2 = lib.ops.random_orthogonal_matrix(X_target.shape[1])
    X_source = X_source @ W1.T
    X_target = X_target @ W2.T

    translations, divergences = lib.adversarial.train_translation(
        X_source, X_target, args.n_instances, 
        batch_size=args.batch_size,
        disc_dim=args.disc_dim,
        lambda_gp=args.lambda_gp,
        lambda_orth=args.lambda_orth,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        print_freq=1000,
        steps=args.steps,
        l2reg_d=args.l2reg_d
    )

    lib.utils.save_image_grid(
        source_pca.inverse(X_source[:100] @ W1), 'source.png')

    for i in range(args.n_instances):
        translated = (X_source[:100] @ translations[i].T)
        translated = target_pca.inverse(translated @ W2)
        lib.utils.save_image_grid(translated, f'translation_{i}.png')

if __name__ == '__main__':
    args = make_args()
    main(args)