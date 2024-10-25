# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch, random, os, wandb
import torch.optim as optim
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

N_CLASSES = 200
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = '.'

assert os.path.exists(f'{ROOT}/tiny-imagenet-200') "download_dataset(AUX_DATA_ROOT) first!"
train_dir = os.path.join(f'{ROOT}/tiny-imagenet-200', 'train')

def seed_everything(seed=42):  
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

#######################################################################################
# Dataset
#######################################################################################

class ValTestDataset(Dataset):
  def __init__(self, img_dir, annotations_file, class_to_idx, transform=None, target_transform=None):
    self.class_to_idx = class_to_idx
    self.idx_to_class = {v: k for k, v in class_to_idx.items()}
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
    if self.transform:
      image = self.transform(image)
      
    label = self.img_labels[img_name]
    if label is not None:
      label = self.class_to_idx[label]
      if self.target_transform:
        label = self.target_transform(label)
    else:
      return image, img_name
    
    return image, label
  
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

TRAIN_DATASET = datasets.ImageFolder(
  root=train_dir, 
  transform=train_transforms, 
  target_transform=target_transform
)

#######################################################################################
# model
#######################################################################################
class SimpNet(nn.Module):
  def __init__(self, n_classes):
    super(SimpNet, self).__init__()
    dims = [66, 128, 192, 288, 355, 432]
    # dims = [33, 64, 96, 144, 177, 216] # HALF

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
  
CRITERION = nn.CrossEntropyLoss()

#######################################################################################
#
#######################################################################################


def get_dataloader(path, kind):
  """
  Return dataloader for a `kind` split of Tiny ImageNet.
  If `kind` is 'val' or 'test', the dataloader should be deterministic.
  path:
    `str`
    Path to the dataset root - a directory which contains 'train' and 'val' folders.
  kind:
    `str`
    'train', 'val' or 'test'

  return:
  dataloader:
    `torch.utils.data.DataLoader` or an object with equivalent interface
    For each batch, should yield a tuple `(preprocessed_images, labels)` where
    `preprocessed_images` is a proper input for `predict()` and `labels` is a
    `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
  """
  seed_everything()

  if kind == 'train':
    dataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=64, shuffle=True, num_workers=NUM_WORKERS)
  elif kind == 'val':
    val_dataset = ValTestDataset(
      img_dir=os.path.join(path, "val/images"), 
      annotations_file=os.path.join(path, "val/val_annotations.txt"), 
      class_to_idx=TRAIN_DATASET.class_to_idx,
      transform=val_transforms,
      target_transform=target_transform
    )
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)
  elif kind == 'test':
    test_dataset = ValTestDataset(
      img_dir=os.path.join(path, "test/images"), 
      annotations_file=None,
      class_to_idx=TRAIN_DATASET.class_to_idx,
      transform=test_transforms,
      target_transform=target_transform
    )
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

  return dataloader
  

def get_model():
  """
  Create neural net object, initialize it with raw weights, upload it to GPU.

  return:
  model:
    `torch.nn.Module`
  """
  model = SimpNet(N_CLASSES).to(DEVICE)
  return model

def get_optimizer(model):
  """
  Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

  return:
  optimizer:
    `torch.optim.Optimizer`
  """
  # I used scheduler, restarts and essembling by 3 epochs, so it is not really a replication
  optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
  return optimizer

@torch.inference_mode()
def predict(model, batch):
  """
  model:
    `torch.nn.Module`
    The neural net, as defined by `get_model()`.
  batch:
    unspecified
    A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
    (with same preprocessing and device).

  return:
  prediction:
    `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
    The scores of each input image to belong to each of the dataset classes.
    Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
    belong to `j`-th class.
    These scores can be 0..1 probabilities, but for better numerical stability
    they can also be raw class scores after the last (usually linear) layer,
    i.e. BEFORE softmax.
  """
  return model(batch) # raw scores

@torch.inference_mode()
def validate(dataloader, model):
  """
  Run `model` through all samples in `dataloader`, compute accuracy and loss.

  dataloader:
    `torch.utils.data.DataLoader` or an object with equivalent interface
    See `get_dataloader()`.
  model:
    `torch.nn.Module`
    See `get_model()`.

  return:
  accuracy:
    `float`
    The fraction of samples from `dataloader` correctly classified by `model`
    (top-1 accuracy). `0.0 <= accuracy <= 1.0`
  loss:
    `float`
    Average loss over all `dataloader` samples.
  """
  total = 0
  correct = 0
  loss = 0
  for inputs, labels in dataloader:
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(inputs)
    loss += CRITERION(outputs, labels).item() * inputs.size(0)
    correct += outputs.argmax(1).eq(labels.argmax(1)).sum().item()
    total += labels.size(0)
  loss /= total
  accuracy = correct / total
  return accuracy, loss
    

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
  """
  Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

  train_dataloader:
  val_dataloader:
    See `get_dataloader()`.
  model:
    See `get_model()`.
  optimizer:
    See `get_optimizer()`.
  """
  for epoch in tqdm(range(100)):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in (train_dataloader):
      inputs = inputs.to(DEVICE)
      labels = labels.to(DEVICE)

      optimizer.zero_grad()
      outputs = model(inputs)
      loss = CRITERION(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * inputs.size(0)
      total += labels.size(0)
      correct += outputs.argmax(1).eq(labels.argmax(1)).sum().item()
      # wandb.log({'train_running_loss': loss.item()/inputs.size(0)})
    
    train_epoch_loss = running_loss / total
    train_epoch_acc = correct / total

    val_epoch_acc, val_epoch_loss = validate(val_dataloader, model)
    # wandb.log({'epoch': epoch + 1,
    # 'val_epoch_acc': val_epoch_acc, 'val_epoch_loss': val_epoch_loss,
    # 'train_epoch_acc': train_epoch_acc, 'train_epoch_loss': train_epoch_loss
    # })

    # # should use wandb for that, i know
    # if epoch % 1 == 0:
    #   checkpoint = {
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': val_epoch_loss
    #   }
    #   torch.save(checkpoint, f'{ROOT}/checkpoints/checkpoint_epoch_{epoch+1}.pth')
   
      

def load_weights(model, checkpoint_path):
  """
  Initialize `model`'s weights from `checkpoint_path` file.

  model:
    `torch.nn.Module`
    See `get_model()`.
  checkpoint_path:
    `str`
    Path to the checkpoint.
  """
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'], strict=True)

def get_checkpoint_metadata():
  """
  Return hard-coded metadata for 'checkpoint.pth'.
  Very important for grading.

  return:
  md5_checksum:
    `str`
    MD5 checksum for the submitted 'checkpoint.pth'.
    On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
    On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
    On Mac, use `$ brew install md5sha1sum`.
  google_drive_link:
    `str`
    View-only Google Drive link to the submitted 'checkpoint.pth'.
    The file must have the same checksum as in `md5_checksum`.
  """
  md5_checksum = 'c81af9622f882ede25a871a71e99db57'
  google_drive_link = "https://drive.google.com/file/d/1cd_OzAVkFFkQ9gU3TBNr9_xnuNy0wea6/view?usp=sharing"

  return md5_checksum, google_drive_link
