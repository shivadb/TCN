import torch
from torchvision import datasets, transforms
import numpy as np
import time


def data_generator(root, batch_size, shuffle=False):
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    datasets.MNIST.resources = [
                                ('/'.join([new_mirror, url.split('/')[-1]]), md5)
                                for url, md5 in datasets.MNIST.resources
                                ]
                                
    train_set = datasets.MNIST(root=root, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
    test_set = datasets.MNIST(root=root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader

def augement_data(data, max_clip, max_pad=100):
    # data expected shape: [batch_size, channels, sequence_length]
    aug_data = torch.zeros(data.size())
    batch_size = aug_data.size()[0]
    seq_len = aug_data.size()[2]
    clip_size = np.random.randint(0, max_clip)
    pad_size = np.random.randint(0, max_pad)

    aug_data[:, :, clip_size+pad_size:] = data[:,:,:seq_len-(clip_size+pad_size)]
    if clip_size > 0:
        aug_data[:, :, :clip_size] = data[torch.randperm(batch_size), :, -clip_size:]

    return aug_data.to(data.device)

def average_runtime(model, num_samples, verbose=True):
    runtimes = []

    for i in range(num_samples):
        if i % 1000 == 0 and verbose: print(f'Finished running {i} samples')

        x = torch.rand((1,1,28*28)).cuda()

        start = time.time()
        model(x)
        runtimes.append(time.time() - start)
    
    return np.mean(runtimes)*1000
