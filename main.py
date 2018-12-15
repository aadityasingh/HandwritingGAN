from mnist import MNIST
import torch

print(torch.cuda.is_availab)

mndata = MNIST('./samples')
mndata.select_emnist('byclass')
print('hi')
mndata.gz=False
images, labels = mndata.load_training()
print('here')
print(len(images))
print(images[0])
print(len(images[0]))
print(len(labels))
print(labels[0])
