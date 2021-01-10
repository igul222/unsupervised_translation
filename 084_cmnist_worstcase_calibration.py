"""
Let's see whether worstcase-over-invariances extracts well-calibrated
predictions given many invariances.
"""

import lib
import torch
import torch.nn.functional as F
from torch import nn, optim

N_INSTANCES = 16
STEPS = 10001

def _loss_wrt_invariance(invariance_logits, classifier_logits):
    """
    invariance_logits: (n_instances, batch_size, n_classes)
    classifier_logits: (n_instances, batch_size, n_classes)
    result: (n_instances,)
    """
    invariance_probs = F.softmax(invariance_logits, dim=-1)
    result = lib.ops.softmax_cross_entropy(classifier_logits, invariance_probs)
    return result.mean(dim=1)

dataset_fn = lib.datasets.colored_mnist
X_source, y_source, X_target, y_target = dataset_fn()
source_pca = lib.pca.PCA(X_source, 128, whiten=True)
target_pca = lib.pca.PCA(X_target, 128, whiten=True)
X_source = source_pca.forward(X_source)
X_target = target_pca.forward(X_target)

# Apply random orthogonal transforms for optimization reasons.
W1 = lib.ops.random_orthogonal_matrix(X_source.shape[1])
W2 = lib.ops.random_orthogonal_matrix(X_target.shape[1])
X_source = X_source @ W1.T
X_target = X_target @ W2.T

source_rep, target_rep, classifier, divergences, target_accs = (
    lib.adversarial.train_dann(
        X_source, y_source, X_target, y_target, N_INSTANCES,
        batch_size=1024,
        detach_Zs=False,
        disc_dim=512,
        steps=STEPS,
        z_dim=32,
        l2reg_r=0.,
        l2reg_c=1e-3,
        l2reg_d=1e-4,
        lambda_erm=1.0,
        lambda_gp=1.0,
        lambda_orth=0.1,
        lr_d=1e-3,
        lr_g=1e-3,
        rep_network='linear'))

Xt = X_target.expand(N_INSTANCES, -1, -1)
invariance_logits = classifier(target_rep(Xt))

n_top = N_INSTANCES# // 2
best = torch.argsort(divergences)[:n_top]
invariance_logits = invariance_logits[best,:,:].clone().detach()

matrix = torch.zeros((n_top, n_top)).cuda()
for i in range(n_top):
    for j in range(n_top):
        matrix[i, j] = _loss_wrt_invariance(
            invariance_logits[i][None,:],
            invariance_logits[j][None,:]
        )[0].detach()
# (i,j)th entry is the cross-entropy of classifier j wrt the
# predictive distribution of classifier i
print('Invariance classifier cross-entropies:')
print(matrix)

classifier = nn.Sequential(
    nn.Linear(X_target.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, int(y_target.max()+1))
).cuda()

def forward():
    classifier_logits = classifier(X_target[None,:,:])
    return _loss_wrt_invariance(invariance_logits, classifier_logits).max()

opt = optim.Adam(classifier.parameters(), weight_decay=1e-3)
lib.utils.train_loop(forward, opt, STEPS)

lib.utils.print_row('prob_geq', 'prob_leq', 'n_examples', 'acc')
preds = F.softmax(classifier(X_target), dim=1)
preds_argmax = preds.argmax(dim=1)
correct = (preds_argmax == y_target).float()
pred_probs = preds[torch.arange(preds.shape[0]).cuda(), preds_argmax]
prob_step = 0.1
for prob_geq in torch.arange(0., 1., step=prob_step):
    prob_leq = prob_geq + prob_step
    mask = (pred_probs >= prob_geq).float()
    mask = mask * (pred_probs <= prob_leq).float()
    n_examples = mask.sum()
    mask_acc = (mask * correct).sum() / n_examples
    lib.utils.print_row(prob_geq, prob_leq, int(n_examples), mask_acc)
    # Save samples
    samples = X_target[torch.nonzero(mask)[:,0]][:100]
    if len(samples) > 0:
        samples = target_pca.inverse(samples @ W2)
        lib.utils.save_image_grid(samples,
            f'samples_{prob_geq.item()}_{prob_leq.item()}.png')
test_acc = lib.ops.multiclass_accuracy(classifier(X_target),y_target).mean()
print('Overall test acc:', test_acc)