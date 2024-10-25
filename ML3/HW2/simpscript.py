import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm

def seed_everything(seed=42):
  import random, os, torch
  import numpy as np

  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

class SimpNet(nn.Module):
  def __init__(self, n_classes):
    super(SimpNet, self).__init__()
    dims = [66, 128, 192, 288, 355, 432]

    self.layers = nn.Sequential(
      nn.Sequential(  # Block 1
        nn.Conv2d(in_channels=3, out_channels=dims[0], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 2
        nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 3
        nn.Conv2d(dims[1], dims[1], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 4
        nn.Conv2d(dims[1], dims[1], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 5
        nn.Conv2d(dims[1], dims[2], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 6
        nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 7
        nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 8
        nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(),
        nn.Dropout(0.2)
      ),
      nn.Sequential(  # Block 9
        nn.Conv2d(dims[2], dims[3], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[3]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.3)
      ),
      nn.Sequential(  # Block 10
        nn.Conv2d(dims[3], dims[3], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[3]),
        nn.ReLU(),
        nn.Dropout(0.3)
      ),
      nn.Sequential(  # Block 11
        nn.Conv2d(dims[3], dims[4], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[4]),
        nn.ReLU(),
        nn.Dropout(0.3)
      ),
      nn.Sequential(  # Block 12
        nn.Conv2d(dims[4], dims[5], kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(dims[5]),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Dropout(0.3)
      )
    )
    self.fc = nn.Linear(dims[5], n_classes)

  def forward(self, x):
    x = self.layers(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

#################################################################################
#################################################################################

class ValTestDataset(Dataset):
  def __init__(self, img_dir, annotations_file, class_to_idx, transform=None, target_transform=None):
    self.class_to_idx = class_to_idx
    self.img_dir = img_dir
    self.img_labels = self._parse_annotations(annotations_file)
    self.transform = transform
    self.target_transform = target_transform
  
  def _parse_annotations(self, annotations_file):
    if annotations_file is not None:
      img_labels = {}
      with open(annotations_file, 'r') as file:
        for line in file:
          img_name, class_name = line.strip().split('\t')[:2]
          img_labels[img_name] = class_name
    else:
      imgs = next(os.walk(self.img_dir))[-1]
      img_labels = {img: None for img in imgs}
    return img_labels
  
  def __len__(self):
    return len(self.img_labels)
  
  def __getitem__(self, idx):
    img_name = list(self.img_labels.keys())[idx]
    img_path = os.path.join(self.img_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    label = self.img_labels[img_name]
    if label is not None:
      label = self.class_to_idx[label]
      if self.target_transform:
        label = self.target_transform(label)
    if self.transform:
      image = self.transform(image)
    
    return image, label


#################################################################################
#################################################################################

wandb.login(relogin=True, key="ac57f7f673f3ffda25cbe4634e094c2b90edbd8f")

seed_everything()

ROOT = 'aim' if os.path.isdir('aim') else '.'
N_CLASSES = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()

target_transform = lambda x: torch.zeros(N_CLASSES).scatter_(0, torch.tensor(x), 1)
train_transforms = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),
  transforms.ColorJitter(),
  transforms.ToTensor(),
  transforms.Normalize([0.480, 0.448, 0.397], [0.230, 0.226, 0.226])
])
val_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.482, 0.449, 0.398], [0.230, 0.226, 0.226])
])
test_transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.471, 0.441, 0.392], [0.230, 0.227, 0.227])
])

img_dir = f'{ROOT}/tiny-imagenet-200/tiny-imagenet-200'
train_dir = f'{img_dir}/train'
val_dir   = f'{img_dir}/val'
test_dir  = f'{img_dir}/test'

train_dataset = datasets.ImageFolder(
  root=train_dir, 
  transform=train_transforms, 
  target_transform=target_transform
)
val_dataset = ValTestDataset(
    img_dir=f"{img_dir}/val/images", 
    annotations_file=f"{val_dir}/val_annotations.txt", 
    class_to_idx=train_dataset.class_to_idx,
    transform=val_transforms,
    target_transform=target_transform
)
test_dataset = ValTestDataset(
    img_dir=f"{test_dir}/images", 
    annotations_file=None,
    class_to_idx=train_dataset.class_to_idx,
    transform=test_transforms,
    target_transform=target_transform
)
  
# DataLoaders
batch_size = 64
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Initialize WandB
wandb.init(project='AIM_SimpNet-TinyImageNet', entity='philurame',
  config={
    'epochs': 100,
    'batch_size': batch_size,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',
    'softmax': 'True...'
  },
  name='with_softmax!!!'
)

config = wandb.config

# Instantiate model
model = SimpNet(N_CLASSES).to(device)

# Loss function and optimizer with L2 regularization (weight decay)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 70], gamma=0.1)

os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)

# Training loop
num_epochs = config.epochs



# checkpoint_path = f'{ROOT}/checkpoints/checkpoint_lr_epoch_41.pth'
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# last_epoch = checkpoint['epoch']
# val_loss = checkpoint['loss']




for epoch in (range(0, num_epochs)):
  if epoch: scheduler.step()

  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for inputs, labels in tqdm(train_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * inputs.size(0)
    total += labels.size(0)
    correct += outputs.argmax(1).eq(labels.argmax(1)).sum().item()

    wandb.log({'train_running_loss': loss.item()/inputs.size(0)})
  
  train_epoch_loss = running_loss / total
  train_epoch_acc = correct / total

  # Validation
  model.eval()
  val_loss = 0.0
  val_correct = 0
  val_total = 0

  with torch.no_grad():
    for val_inputs, val_labels in val_loader:
      val_inputs = val_inputs.to(device)
      val_labels = val_labels.to(device)

      val_outputs = model(val_inputs)
      loss = criterion(val_outputs, val_labels)
      val_loss += loss.item() * val_inputs.size(0)
      val_total += val_labels.size(0)
      val_correct +=  val_outputs.argmax(1).eq(val_labels.argmax(1)).sum().item()

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total

    wandb.log(
      {
        'epoch': epoch + 1, 'lr': scheduler.get_last_lr()[0],
        'train_epoch_loss': train_epoch_loss, 'train_epoch_accuracy': train_epoch_acc,
        'val_epoch_loss':   val_epoch_loss, 'val_epoch_accuracy': val_epoch_acc
      }
    )

    if epoch % 1 == 0:
      # Save checkpoint
      checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_epoch_loss
      }
      torch.save(checkpoint, f'{ROOT}/checkpoints/checkpoint_sm_epoch_{epoch+1}.pth')