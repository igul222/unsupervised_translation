"""
Different methods for combining many DANN-style invariances into a single
predictor.
"""

import collections
import lib
import torch
import torch.nn.functional as F
from torch import nn, optim

def _loss_wrt_invariance(invariance_logits, classifier_logits):
    """
    invariance_logits: (n_instances, batch_size, n_classes)
    classifier_logits: (n_instances, batch_size, n_classes)
    result: (n_instances,)
    """
    invariance_probs = F.softmax(invariance_logits, dim=-1)
    result = lib.ops.softmax_cross_entropy(classifier_logits, invariance_probs)
    return result.mean(dim=1)

def top_4(
    classifier,
    divergences,
    target_rep,
    X_target,
    y_target):
    """
    Return the mean aaccuracy of the top 4 classifiers (by divergence).
    """
    with torch.no_grad():
        n_instances = len(divergences)
        best = torch.argsort(divergences)[:4]
        Xt = X_target.expand(n_instances, -1, -1)
        logits = classifier(target_rep(Xt))[best]
        return lib.ops.multiclass_accuracy(logits, y_target).mean()

def random(
    classifier,
    divergences,
    target_rep,
    X_target,
    y_target):
    """
    Return the expected accuracy of picking an invariance randomly (i.e. the
    mean accuracy across all classifiers).
    """
    with torch.no_grad():
        n_instances = len(divergences)
        Xt = X_target.expand(n_instances, -1, -1)
        logits = classifier(target_rep(Xt))
        return lib.ops.multiclass_accuracy(logits, y_target[None,:]).mean()

def worstcase(
    classifier,
    divergences,
    target_rep,
    X_target,
    y_target):
    """
    Return the accuracy of a classifier trained to minimize worst-case error
    over the top half of invariances (by divergence).
    """
    with torch.no_grad():
        n_instances = len(divergences)

        Xt = X_target.expand(n_instances, -1, -1)
        invariance_logits = classifier(target_rep(Xt))

        n_top = n_instances // 2
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

    opt = optim.Adam(classifier.parameters())
    lib.utils.train_loop(forward, opt, 10001)

    test_acc = lib.ops.multiclass_accuracy(classifier(X_target),y_target).mean()
    return test_acc

def expectation(
    classifier,
    divergences,
    target_rep,
    X_target,
    y_target):
    """
    Return the accuracy of the classifier which averages the predictions of
    all of the invariance classifiers.
    """
    with torch.no_grad():
        n_instances = len(divergences)
        Xt = X_target.expand(n_instances, -1, -1)
        preds = F.softmax(classifier(target_rep(Xt)), dim=2)
        return lib.ops.multiclass_accuracy(preds.mean(dim=0), y_target).mean()

REGISTRY = collections.OrderedDict([
    ('random', random),
    ('top_4', top_4),
    ('worstcase', worstcase),
    ('expectation', expectation)
])