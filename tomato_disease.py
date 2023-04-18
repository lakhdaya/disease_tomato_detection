import torch
import os
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime


from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 10)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(training_loader), total = len(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data.to('cuda')
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def train_epochs(model, loss_fn, optimizer, training_loader, testing_loader, writer, timestamp):
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in tqdm(range(EPOCHS)):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, training_loader, model, optimizer, loss_fn)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(testing_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}\n'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def main(model_name):

    data_path = os.path.realpath("PlantVillage")
    labels_name = [name[0].split("\\")[-1]  for name in os.walk(data_path) 
                if "Tomato" in name[0]]
    directory_paths = [os.path.join(data_path, name) for name in labels_name]

    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights="DenseNet121_Weights.DEFAULT")

    data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    dataset = ImageFolder(root=data_path, transform=data_transforms)


    # move the input and model to GPU for speed if available
    print("{} is available".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("gpu used")   
        model.to('cuda')
    train_size = int(0.66 * len(dataset))
    test_size = len(dataset) - train_size

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    training_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    testing_loader = DataLoader(test_set, batch_size=10, shuffle=True)
    # Initializing in a separate cell so we can easily add more epochs to the same run
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/plant_trainer_{}'.format(timestamp))
    
    train_epochs(model, loss_fn, optimizer, training_loader, testing_loader, writer, timestamp)
    torch.save("models/{}".format(model_name), model)

def test_model(model):
    None

if __name__ == "__main__":
    model_name = "test_model"
    main(model_name)