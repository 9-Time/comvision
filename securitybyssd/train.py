from dataloader.openimagesloader import *
from dataloader.transformations import *

import torch
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import time

from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.config import vgg_ssd_config
from vision.ssd.ssd import MatchPrior
from vision.nn.multibox_loss import MultiboxLoss

DATASET_DIRECTORY = 'data/open_images'
CHECKPOINT_DIRECTORY = 'checkpoint'
if not os.path.exists(CHECKPOINT_DIRECTORY):
    os.makedirs(CHECKPOINT_DIRECTORY)
learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
batch_size = 8
gamma = 0.1
num_epochs = 30
num_workers = 0
debug_steps = 100

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    print('Using CUDA')

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    epoch_loss = 0.0
    num = 0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.long()
        labels = labels.to(device)
        
        num += 1
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        epoch_loss += running_loss
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            print(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f} , " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
    return epoch_loss/num

def val(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.long()
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

if __name__ == '__main__':
    ############# NEEDS EDITING #############
    create_net = create_vgg_ssd
    config = vgg_ssd_config
    #########################################

    train_transform = TrainAugmentation(300)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(300)
    print('Preparing Training Dataset')
    train_dataset = OpenImageData(DATASET_DIRECTORY, train_val_test ='train',
            transform=train_transform, target_transform=target_transform)
    with open(CHECKPOINT_DIRECTORY+'/open-images-model-labels.txt', "w") as f:
        f.write("\n".join(train_dataset.class_names))
    num_classes = len(train_dataset.class_names)
    print("Stored labels into file")
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    print("# Train Data:{}".format(len(train_dataset)))

    print('Preparing Validation Dataset')
    val_dataset = OpenImageData(DATASET_DIRECTORY, train_val_test ='validation',
            transform=train_transform, target_transform=target_transform)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=num_workers, shuffle=False)
    print("# Validation Data:{}".format(len(val_dataset)))
    ############# NEEDS EDITING #############
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    #########################################
    net.to(device)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)

    milestones = [80,100]
    scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                    gamma=0.1, last_epoch=last_epoch)
    print("Starting Training")
    start_time = time.time()
    train_losses = []
    val_losses = []
    val_reg_losses = []
    val_class_losses = []
    for epoch in range(1, num_epochs+1):
        scheduler.step()
        epoch_trainloss = train(train_loader, net, criterion, optimizer,
                                device=device, debug_steps=debug_steps, epoch=epoch)
        train_losses.append(epoch_trainloss)
        train_end = time.time() - start_time
        train_sec = train_end//60
        train_min = train_end%60
        train_hour = train_min%60
        train_min = train_min//60
        print(f'{train_hour}h {train_min}min {train_sec}s Epoch {epoch}, Train Loss: {epoch_trainloss:.4f}')
        val_loss, val_regression_loss, val_classification_loss = val(val_loader, net, criterion, device)
        val_losses.append(val_loss)
        val_reg_losses.append(val_regression_loss)
        val_class_losses.append(val_classification_loss)
        val_end = time.time() - start_time
        val_sec = val_end//60
        val_min = val_end%60
        val_hour = val_min%60
        val_min = val_min//60
        print(f"{val_hour}h {val_min}min {val_sec}s Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
                )
        model_path = os.path.join(CHECKPOINT_DIRECTORY, f"vgg-Epoch-{epoch}-Loss-{val_loss}.pth")
        net.save(model_path)
        print(f"Saved model {model_path}")
        with open(CHECKPOINT_DIRECTORY+'/epoch{}.txt'.format(epoch), "w") as f:
            f.write('trainloss='+",".join(train_losses))
            f.write('valloss='+",".join(val_losses))
            f.write('valregloss='+",".join(val_reg_losses))
            f.write('valclassloss='+",".join(val_class_losses))