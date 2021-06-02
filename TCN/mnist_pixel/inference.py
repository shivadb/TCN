import torch
import torch.nn.functional as F
import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from TCN.mnist_pixel.utils import data_generator
from TCN.mnist_pixel.model import TCN
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--modelpath', type=Path, default='./models/default_params.pt',
                    help='directory to save model')
parser.add_argument('--datapath', type=Path, default='./data/mnist',
                    help='directory to save model')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

_, test_loader = data_generator(args.datapath, args.batch_size)
model = TCN()
model.load_state_dict(torch.load(args.modelpath))
model.eval()

if args.cuda:
    model.cuda()

model.fast_inference(args.batch_size)

def test():
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(data.size()[0], 1, -1)
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

def single_test():
    orig_loss = 0
    single_loss = 0
    orig_correct = 0
    single_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(data.size()[0], 1, -1)
             # print(model.compare(data))
            orig_out = model(data)
            for i in range(data.size()[2]):
                single_out = model.single_forward(data[:,:,i].view(data.size()[0], data.size()[1], 1))
            
            model.reset_cache()
            print((single_out == orig_out).all().item())

            orig_loss += F.nll_loss(orig_out, target, reduction='sum').item()
            single_loss += F.nll_loss(single_out, target, reduction='sum').item()
            orig_pred = orig_out.max(1, keepdim=True)[1]
            single_pred = single_out.max(1, keepdim=True)[1]
            orig_correct += orig_pred.eq(target.data.view_as(orig_pred)).cpu().sum()
            single_correct += single_pred.eq(target.data.view_as(single_pred)).cpu().sum()

        orig_loss /= len(test_loader.dataset)
        single_loss /= len(test_loader.dataset)

        print('\nTest set: Original Average loss: {:.4f}, Original Accuracy: {}/{} ({:.2f}%)\n'.format(
            orig_loss, orig_correct, len(test_loader.dataset),
            100. * orig_correct / len(test_loader.dataset)))
        print()
        print('\nTest set: Single Average loss: {:.4f}, Single Accuracy: {}/{} ({:.2f}%)\n'.format(
            single_loss, single_correct, len(test_loader.dataset),
            100. * single_correct / len(test_loader.dataset)))

        return orig_loss, orig_loss


def timing_tests():
    with torch.no_grad():
        orig_times = []
        single_times = []
        # orig_out = model(torch.zeros(args.batch_size, 1, 784).cuda()) # first run takes significantly longer
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(data.size()[0], 1, -1)
            
            start = time.time()
            orig_out = model(data)
            orig_times.append(time.time()-start)

            total_single_time = 0
            for i in range(data.size()[2]):
                single_data = data[:,:,i].view(data.size()[0], data.size()[1], 1)
                start = time.time()
                single_out = model.single_forward(single_data)
                total_single_time += time.time()-start

            

            single_times.append(total_single_time/data.size()[2])

            print(f'Original runtime: {orig_times[-1]*1000}ms')
            print(f'Single runtime: {single_times[-1]*1000}ms')
            
    print(f'Average original runtime: {np.array(orig_times).mean()*1000}ms')
    print(f'Average single runtime: {np.array(single_times).mean()*1000}ms')


if __name__ == "__main__":
    test()
    # single_test()
    # timing_tests()