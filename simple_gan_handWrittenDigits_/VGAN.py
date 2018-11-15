#https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
#%reload_ext autoreload
#%matplotlib inline

from IPython import display
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import os
os.chdir('/home/carlos/Python/Torch/GAN')

from utils import Logger


def mnist_data():
    compose = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
    out_dir = '{}/dataset'.format(os.getcwd())
    return datasets.MNIST(root=out_dir, train=True,
                          transform=compose, download=True)
    
#Load data
data = mnist_data()
    
#Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size = 100,
                                              shuffle=True)
    
#Num batches
num_batches = len(data_loader)
    
  
"""
Define Generator and Discriminator classes
"""

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
        )
        self.hidden2= nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
                nn.Linear(256, n_out),
                nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
        
discriminator = DiscriminatorNet()
"""
 additional functionality that allows us to convert a flattened image into its 2-dimensional representation, 
 and another one that does the opposite
"""

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)
  

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    Generative Network takes a latent variable vector as 
    input, and returns a 784 valued vector, which corresponds
    to a flattened 28x28 image. Remember that the purpose of 
    this network is to learn how to create undistinguishable 
    images of hand-written digits,which is why its output is 
    itself a new image. The output layer will have a TanH 
    activation function, which maps the resulting values into
    the (-1, 1) range, which is the same range in which our
    preprocessed MNIST images is bounded
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential( 
                nn.Linear(n_features, 256),
                nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
                nn.Linear(1024, n_out),
                nn.Tanh()
        )
        
    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

generator = GeneratorNet()
    
"""
Define Optimization and Loss
"""
d_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr = 0.0002)

loss = nn.BCELoss()
"""
Mean is calculated by computing sum(L) / N

Binary Cross Entopy Loss (BCE Loss), 
it will be used for this scenario as it resembles the 
log-loss for both the Generator and Discriminator 
defined earlier in the post. 
Specifically we’ll be taking the average of the loss 
calculated for each minibatch.

The real-images targets are always ones, 
while the fake-images targets are zero, so it would be 
helpful to define the following functions
"""
"""
We also need some additional functionality that allows us to
 create the random noise. The random noise will be sampled 
 from a normal distribution with mean 0 and variance 1
"""

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n


def real_data_target(size):
    data = Variable(torch.ones(size, 1))
    return data

def fake_data_target(size):
    data = Variable(torch.zeros(size,1))
    return data

def train_discriminator(optimizer, real_data, fake_data):
    #Reset gradients
    optimizer.zero_grad()
    
    #1.1 Train on real data
    prediction_real = discriminator(real_data)
    #Calculate error and back propagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()
    
    #1.2 Train on fake data
    prediction_fake = discriminator(fake_data)
    #Calculate error and back propagate
    error_fake = loss(prediction_fake, fake_data_target(fake_data.size(0)))
    error_fake.backward()
    
    #1.3 Update weights with gradients
    optimizer.step()
    
    #Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    #reset gradient
    optimizer.zero_grad()
    
    #Sample noise and generate fake data
    prediction = discriminator(fake_data)
    
    #error
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    
    #Update weights with gradients
    optimizer.step()
    return error

'''
Testing
We want to visualize how the training process develops while our GAN learns. 
To do so, we will create a static batch of noise, every few steps we will 
visualize the batch of images the generator outputs when using this noise as input.
'''
num_test_samples = 16
test_noise = noise(num_test_samples)

'''
Training
'''
#Create logger instance
logger = Logger(model_name='GAN', data_name='MNIST')
#Num of epochs
num_epochs = 20
#trian discriminator this many times before training generator
d_steps = 4 

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
                        
    #1.0 Train discriminator
        for step in range(d_steps):
            
            real_data = Variable(images_to_vectors(real_batch))
            
            #Generate fake data and detach
            #so gradients are not calculated for generator
            fake_data = generator(noise(real_data.size(0))).detach()
            
            #Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        
    #2.0 Train Generator
        #Generate fake data
        fake_data = generator(noise(real_batch.size(0)))
        
        #Train G
        g_error = train_generator(g_optimizer, fake_data)
        
    #3.0 Logging
        #Log batch error    
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        
        #Display progress every few batches
        if (n_batch) % 100 == 0:
            #Display images
            display.clear_output(True)
            test_images = vectors_to_images(generator(test_noise)).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
            #Display status logs
            logger.display_status(epoch, num_epochs, n_batch, num_batches,
                                          d_error, g_error, d_pred_real, d_pred_fake)
        #Model checkpoints
        logger.save_models(generator, discriminator, epoch)
                    
            
        
        
        