from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data
from models import GCN
from sklearn.metrics import roc_curve, auc

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=4,
                    help='class-balanced parameter.')
parser.add_argument('--beta', type=float, default=1e-2,
                    help='l2 regularization parameter).')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_test, idx_val = load_data("cora")  #cora, citeseer, pubmed

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    
weights=[args.alpha, 1]
class_weights = torch.FloatTensor(weights)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, w1, w2, w3, w4 = model(features, adj)
    
    w1 = torch.pow(torch.norm(w1), 2)
    w2 = torch.pow(torch.norm(w2), 2)
    w3 = torch.pow(torch.norm(w3), 2)
    w4 = torch.pow(torch.norm(w4), 2)
    l2_reg = w1 + w2 + w3 + w4 

    loss = torch.nn.CrossEntropyLoss(weight=class_weights)    
    loss_train = loss(output[idx_train], labels[idx_train]) + args.beta*l2_reg
    loss_train.backward()
    optimizer.step()
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, w1, w2, w3, w4 = model(features, adj) 
        
    loss_val = loss(output[idx_val], labels[idx_val]) + args.beta*l2_reg    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output, _, _, _, _ = model(features, adj)

    loss = torch.nn.CrossEntropyLoss(weight = class_weights)
    loss_test = loss(output[idx_test], labels[idx_test])

    scores = F.softmax(output, dim=1)
    fpr, tpr, t = roc_curve(labels[idx_test].detach().numpy(), scores[idx_test,0].detach().numpy(), pos_label = 0)
    roc_auc= auc(fpr, tpr)    
    return roc_auc

# Train model
t_total = time.time()
max_auc = 0
for epoch in range(args.epochs):
    train(epoch)
    roc_auc = test()
    if roc_auc>max_auc:
        max_auc = roc_auc
AUC = max_auc
print('AUC: {:04f}'.format(AUC))
