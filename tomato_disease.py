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

def train_one_epoch(epoch_index, tb_writer, training_loader, model, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(training_loader), total = len(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
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

    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    dataset = ImageFolder(root=data_path, transform=data_transforms)

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