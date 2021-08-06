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

def augment_data(data, max_clip, max_pad=100):
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

def average_runtime(model, num_samples, single_mode=False, fp16=False, verbose=True):
    runtimes = []

    for i in range(num_samples):
        if i % 1000 == 0 and verbose: print(f'Finished running {i} samples')

        x = torch.rand((1,1,28*28)).cuda() if not single_mode else torch.rand((1,1,1)).cuda()
        x = x.half() if fp16 else x

        start = time.time()
        model(x)
        runtimes.append(time.time() - start)
    
    return np.mean(runtimes)*1000


def predict_image(model, data, data_queue=None, num_samples=100):
    data_queue = torch.zeros((1,1,28*28)).to(data.device) if data_queue is None else data_queue
    data = data.view(data.size()[0], 1, -1) #flatten image
    data_queue = torch.cat((data_queue[:, :, -num_samples:], data[:, :, :-num_samples]), dim=2)

    predictions = []
    for i in range(num_samples):
        data_queue = data_queue.roll(-1,2)
        data_queue[:, :, -1] = data[:, :, -num_samples+i]
        predictions.append(model(data_queue).item())
    
    # return the most frequent prediction
    return np.bincount(predictions).argmax()
    

def model_accuracy(model, data_loader, num_samples=100, fp16=False, verbose=True):
    data_queue = torch.zeros((1,1,28*28))
    data_queue = data_queue.cuda() if torch.cuda.is_available() else data_queue
    data_queue = data_queue.half() if fp16 else data_queue

    predictions = []
    for i, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        data = data.half() if fp16 else data


        prediction = predict_image(model, data, data_queue, num_samples)
        predictions.append(prediction == target.item())

        if verbose and i % 50 == 0: 
            print(f'Accuracy after {i+1} images: {np.sum(predictions)/len(predictions)}')
    
    return np.sum(predictions)/len(predictions)
