import numpy as np
import os
import sys

dataset_dir = '/HybridCiM/data/test/dataset'
dataset_raw_dir = '/HybridCiM/data/test/dataset/raw'

def create_input(data, name):
    data_list = []
    for i in range(len(data)):
        temp_inp = {'data': data[i], 'counter': 100 * np.ones(len(data[i])), 'valid': np.ones(len(data[i]))}
        data_list.append(temp_inp)
    
    np.save(dataset_dir + '/' + name + '.npy', data_list)


def create_mnist_dataset():
    mnist_dir = os.path.join(dataset_raw_dir, 'MNIST')

    try:
        # Load local MNIST data
        with open(os.path.join(mnist_dir, 'train-images.idx3-ubyte'), 'rb') as f:
            f.read(16)  # Skip header
            images = np.frombuffer(f.read(), dtype=np.uint8)
            x_train = images.reshape(-1, 784).astype('float32') / 255

        with open(os.path.join(mnist_dir, 'train-labels.idx1-ubyte'), 'rb') as f:
            f.read(8)  # Skip header
            y_train = np.frombuffer(f.read(), dtype=np.uint8)

        with open(os.path.join(mnist_dir, 't10k-images.idx3-ubyte'), 'rb') as f:
            f.read(16)  # Skip header
            images = np.frombuffer(f.read(), dtype=np.uint8)
            x_test = images.reshape(-1, 784).astype('float32') / 255

        with open(os.path.join(mnist_dir, 't10k-labels.idx1-ubyte'), 'rb') as f:
            f.read(8)  # Skip header
            y_test = np.frombuffer(f.read(), dtype=np.uint8)

        # Create output directory
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Save processed data
        create_input(x_train, 'mnist_train_input')
        create_input(x_test, 'mnist_test_input')
        np.save(dataset_dir + '/mnist_train.npy', x_train)
        np.save(dataset_dir + '/mnist_train_labels.npy', y_train)
        np.save(dataset_dir + '/mnist_test.npy', x_test)
        np.save(dataset_dir + '/mnist_test_labels.npy', y_test)
        print(("Successfully created MNIST dataset at: {}".format(dataset_dir)))
        
    except IOError as e:
        print(("Error: MNIST files not found in", mnist_dir))
        print("Please download MNIST dataset from http://yann.lecun.com/exdb/mnist/")
        print(("and extract to", mnist_dir))
        sys.exit(1)

if __name__ == '__main__':
    create_mnist_dataset()