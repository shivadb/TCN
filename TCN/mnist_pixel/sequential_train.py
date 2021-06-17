import torch
import torch.optim as optim
import torch.nn.functional as F
import os, sys
from pathlib import Path
# sys.path.append("..\\..\\")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from TCN.mnist_pixel.utils import data_generator
from TCN.mnist_pixel.model import TCN
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
parser.add_argument('--savedir', type=Path, default='./models',
                    help='directory to save model')
parser.add_argument('--savemodel', action='store_true',
                    help='save best model (default: false)')
parser.add_argument('--modelname', default='model',
                    help='name used to save trained model')
parser.add_argument('--trainsequential', action='store_true',
                    help='train by using single pixel inputs')
args = parser.parse_args()

# python sequential_train.py --ksize 7 --levels 6 --modelname seq_test --trainsequential

if args.seed > 0:
    torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

if args.trainsequential:
    model.fast_inference(args.batch_size)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_loss = 0
    epoch_loss = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f'Batch: {batch_idx}')
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]

        optimizer.zero_grad()

        seq_loss = 0
        if args.trainsequential:
            for i in range(data.size()[2]):
                print(i)
                single_out = model.single_forward(data[:,:,i].view(data.size()[0], data.size()[1], 1))
                if i > model.receptive_field//2 and np.random.randint(50) == 0:
                    loss = F.nll_loss(single_out, target)
                    loss.backward(retain_graph=True)
                    seq_loss += loss
        else:
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if args.trainsequential:
            train_loss += seq_loss
            epoch_loss.append(seq_loss.item())
        else:
            train_loss += loss
            epoch_loss.append(loss.item())
        
        steps += seq_length

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0
        
        for i in np.zeros(np.random.randint(50, 200)):
            output = model.single_forward(torch.tensor([i]*batch_size, dtype=torch.float).cuda().view(batch_size, 1, 1))


    return np.array(epoch_loss).mean()

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            # data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":
    if args.savemodel:
        fname = args.modelname + '_args.pkl'
        pickle.dump(args, open(args.savedir / fname, 'wb'))
    

    for epoch in range(1, epochs+1):
        train_loss = train(epoch)
        lowest_test_loss = None
        test_loss = test()

        if args.savemodel and (not lowest_test_loss or test_loss < lowest_test_loss):
            fname = args.modelname if args.modelname.endswith('.pt') or args.modelname.endswith('.pth') else args.modelname + '.pt'
            torch.save(model.state_dict(), args.savedir / fname)

        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
