
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import glob
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

from sklearn.model_selection import KFold
from Model_Final_Precleanup import ResNet
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import matplotlib.image as img
# import random

from torch.optim import lr_scheduler
import time
import os
import copy



print('START')
TRAIN_CSV_PATH = r'./dataset/train_labels.csv'
TEST_CSV_PATH = r'./dataset/sample.csv'
TRAIN_PATH = r'./dataset/train_set/train_set/'
TEST_PATH = r'./dataset/test_set/test_set/'

labels = pd.read_csv(TRAIN_CSV_PATH)

train_df = pd.read_csv(TRAIN_CSV_PATH) # labels should start at 0.
train_df['label'] = train_df['label'] - 1

test_df = pd.read_csv(TEST_CSV_PATH) # labels should start at 0.
test_df['label'] = test_df['label'] - 1



# Hyperparameters
N_CLASSES = 80
BATCH_SIZE = 16
IN_CHANNELS = 3
IMG_SIZE = 256
TRANSFER_LEARNING = True
PROB = 0.5 # Probability of random data augmentations

RESIZES = [144, 144, 256, 256]
CROP_SIZES = RESIZES  # [128, 128, 224, 224]
EPOCHS_PER_SIZE = [3, 3, 3, 3]
LEARNING_RATES = [3e-4, 1e-4, 3e-4, 3e-5]

means = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


SPLIT = 0.8
N_TRAIN = round(SPLIT * len(train_df))

N_VAL = round(len(train_df) - N_TRAIN)

print("Train set length: ", N_TRAIN,"\n",
      "Validation set length: ", N_VAL)



loss_func = nn.CrossEntropyLoss().to(device)
model = ResNet(N_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATES[0])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')


class FoodDataset(Dataset):
    """Custom Dataset for loading images"""
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = df.img_name.values
        self.y = df.label.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


def get_train_loader(img_size, crop_size): # OBSOLETE

    train_transforms = transforms.Compose([
                                       transforms.Resize((img_size, img_size)),
                                       #transforms.RandomRotation(5),
                                       transforms.RandomHorizontalFlip(PROB),
                                        # transforms.GaussianBlur(7, sigma=(0.2, 0.3)), # kernel size = 7, used std
                                      #  transforms.RandomPerspective(distortion_scale=0.5, p=PROB,
                                      #  fill=0), # Interpolation mode left to default bilinear default left for distrotion
                                      transforms.ColorJitter(brightness=0.449, contrast=0.449, saturation=0.449, hue=0.449), # mean values used
                                      # transforms.RandomErasing(p=PROB, scale=(0.8,1), inplace=False), #scale set to resize mean with random uper bound (error margin should be used!)
                                       transforms.RandomCrop((crop_size, crop_size)),
                                        # transforms.functional.adjust_hue(hue_factor=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, std),
                                      ]
                                    )

    # Train/Validation splits

    train_dataset = FoodDataset(df=train_df, 
                                img_dir=TRAIN_PATH[:N_TRAIN],
                                transform=train_transforms
                            )
    
    val_dataset =  FoodDataset(df=train_df,
                                img_dir=TRAIN_PATH[N_TRAIN:],
                                transform=False
                            )




      return train_loader, val_loader


def get_test_loader(img_size, crop_size):

    test_transforms = transforms.Compose([
                                            transforms.Resize((img_size, img_size)),
                                            transforms.RandomCrop((crop_size, crop_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means, std),
                                            ]
                                        )

    test_dataset = FoodDataset(df=test_df,
                                img_dir=TEST_PATH,
                                transform=test_transforms
                            )

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True 
                            )

    return test_loader


def train(train_loader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Mixup augmentation
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))


        outputs, _ = model(inputs)
        loss = mixup_criterion(loss_func, outputs, targets_a, targets_b, lam)


        # loss = loss_func(outputs, targets)   

        optimizer.zero_grad()             
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    

def predict(test_loader):
    model.eval()
    predicted_labels = np.array([])
    i = 0
    for data in test_loader:

        if i % 1000 == 999: 
            print(len(predicted_labels))
        i += 1

        input, target = data
        input = input.to(device)
        target = target.to(device)

        # forward + backward + optimize
        outputs, _ = model(input)

        prediction = outputs.argmax(dim=1).detach().cpu().numpy()
        prediction = prediction + 1 # Correct label for earlier transformation.
        predicted_labels = np.append(predicted_labels, prediction)

    print('Finished Predictions')
    return predicted_labels


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def change_lr(lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


if __name__ == '__main__':

    if TRANSFER_LEARNING:
        print('TRANSFER WEIGHTS')
        pretrained_model = models.resnet50(pretrained=True)
        IN_FEATURES = pretrained_model.fc.in_features 
        pretrained_model.fc = nn.Linear(IN_FEATURES, N_CLASSES)
        print(model.load_state_dict(pretrained_model.state_dict()))

    print('\nBEGIN TRAINING')

    # Progressive resizing
    for epochs, img_size, lr, crop_size in zip(EPOCHS_PER_SIZE, RESIZES, LEARNING_RATES, CROP_SIZES):
        print(f'\nimage size: {img_size}')


        train_transforms = transforms.Compose([
                                       transforms.Resize((img_size, img_size)),
                                       #transforms.RandomRotation(5),
                                       transforms.RandomHorizontalFlip(PROB),
                                        # transforms.GaussianBlur(7, sigma=(0.2, 0.3)), # kernel size = 7, used std
                                      #  transforms.RandomPerspective(distortion_scale=0.5, p=PROB, # Best at 0.64
                                      #  fill=0), # Interpolation mode left to default bilinear default left for distrotion
                                      # transforms.ColorJitter(brightness=0.449, contrast=0.449, saturation=0.449, hue=0.449), # mean values used
                                      # transforms.RandomErasing(p=PROB, scale=(0.8,1), inplace=False), #scale set to resize mean with random uper bound (error margin should be used!)
                                       transforms.RandomCrop((crop_size, crop_size)),
                                        # transforms.functional.adjust_hue(hue_factor=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, std),
                                      ]
                                    )

        # Train/Validation splits

        train_dataset = FoodDataset(df=train_df, 
                                    img_dir=TRAIN_PATH[:N_TRAIN],
                                    transform=train_transforms
                                )
        
        val_dataset =  FoodDataset(df=train_df,
                                    img_dir=TRAIN_PATH[N_TRAIN:],
                                    transform=False
                                )


        # K fold cross validation:

        k_fold = KFold(n_splits=10,shuffle=True) # set to stadard 10 splits 

        for fold,(train_index,val_index) in enumerate(k_fold.split(train_dataset)):

          print('------------fold no---------{}----------------------'.format(fold))

          train_sub = torch.utils.data.SubsetRandomSampler(train_index)
          val_sub = torch.utils.data.SubsetRandomSampler(val_index)

          train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=BATCH_SIZE,
                                      # shuffle=True,
                                      pin_memory=True,
                                      sampler=train_sub
                                  )

          val_loader = DataLoader(dataset=val_dataset,
                                  batch_size=BATCH_SIZE,
                                  # shuffle=True,
                                  pin_memory=True,
                                  sampler=val_sub
                              )



        # train_loader = get_train_loader(img_size, crop_size)[0]
        
          change_lr(lr)
          print(f"\nlearning rate set to {optimizer.param_groups[0]['lr']}")

          for epoch in range(epochs):
              train(train_loader, epoch)

    print('BEGIN PREDICTION')
    val_loader = get_train_loader(img_size, crop_size)[1] # fka get_test_loader
    predicted_labels = predict(val_loader) #fka test_loader

    # Write results to csv file
    results = test_df.copy(deep=True)
    results['label'] = predicted_labels.tolist()
    results['label'] = results['label'].astype(int)
    results.to_csv('results.csv', index=False) 
    files.download('results.csv')

    torch.cuda.empty_cache() 

