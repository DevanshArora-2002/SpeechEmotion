import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import timm
import torch
class datasets(Dataset):
  def __init__(self,paths,ids):
    self.paths=paths
    self.id=ids
  def __len__(self):
    return len(self.paths)
  def __getitem__(self,idx):
    audio,sr=librosa.load(self.paths.iloc[idx],sr=22050)
    return torch.tensor(create_spectrogram(audio,sr)),torch.tensor(self.id.iloc[idx])
class model(nn.Module):
  def __init__(self,config,ckpt):
    super().__init__()
    self.config=config
    self.model=timm.create_model(ckpt,pretrained=True)
    for param in self.model.parameters():
      param.requires_grad=True
    self.model_config=self.model.default_cfg
    self.hidden=self.model_config['num_classes']
    self.classifier=nn.Sequential(
        nn.Linear(self.hidden,self.hidden),
        nn.Dropout(0.2),
        nn.Linear(self.hidden,self.config['num_labels'])
    )
  def freeze_parameters(self):
    for param in self.model.parameters():
      param.requires_grad=False
  def forward(self,input,labels=None):
    x=input
    x=self.model(x)
    x=self.classifier(x)
    if(labels is not None):
      loss_fn=nn.CrossEntropyLoss()
      return x,loss_fn(x,labels)
    return x
def normalize_coordinates(img):
    image_size=img.shape
    height, width = image_size

    # Create a grid of x and y coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize the coordinates in the range of -1 to 1
    normalized_x = ((x_coords / (width - 1)) - 0.5) * 2
    normalized_y = ((y_coords / (height - 1)) - 0.5) * 2

    # Stack the normalized x and y coordinates into a single array
    normalized_coords = np.stack([normalized_x, normalized_y], axis=-1)
    img=np.reshape(img,(img.shape[0],img.shape[1],1))
    normalized_coords= np.concatenate([img,normalized_coords],axis=-1)
    width,height,channels=normalized_coords.shape
    normalized_coords=np.reshape(normalized_coords,(channels,width,height))
    return normalized_coords
def create_spectrogram(data, sr):
    # Compute the spectrogram
    X = librosa.stft(y=data, n_fft=1024, hop_length=128, win_length=1024)
    Xdb = librosa.amplitude_to_db(abs(X))

    # Retrieve the corresponding time axis
    t = librosa.frames_to_time(range(Xdb.shape[1]), sr=sr, hop_length=128)

    # Retrieve the corresponding frequency axis
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

    # Plot the spectrogram
    # plt.figure(figsize=(12, 3))
    # plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='linear', hop_length=128)
    # plt.colorbar(format="%+2.0f dB")
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    spec=Xdb
    min=np.min(spec)
    max=np.max(spec)
    spec=spec-min
    spec=spec/(max-min)
    spec=spec*255
    spec=np.ceil(spec)
    spec2=cv2.resize(spec,(224,224))
    return normalize_coordinates(spec2)

from sklearn.metrics import accuracy_score,precision_score,recall_score
def evaluate(predictions,ground_truth):
  predictions=np.array(predictions.detach().to('cpu'))
  ground_truth=np.array(ground_truth.detach().to('cpu'))
  predictions=np.argmax(predictions,axis=1)
  acc=accuracy_score(predictions,ground_truth)
  prec=precision_score(predictions,ground_truth,average='weighted')
  rec=recall_score(predictions,ground_truth,average='weighted')
  return {
      'accuracy':acc,
      'precision':prec,
      'recall':rec
  }

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def train_model(model,loader,val_loader,optim,epoch):
  loss_curve=[]
  acc_curve=[]
  rec_curve=[]
  prec_curve=[]

  val_loss_curve=[]
  val_acc_curve=[]
  val_rec_curve=[]
  val_prec_curve=[]
  device=model.device
  model.train()
  for ep in range(epoch):
    training_loss=0.0
    acc=0
    recall=0
    prec=0
    for st,(input,labels) in tqdm(enumerate(loader)):
      optim.zero_grad()
      input=input.to(torch.float).to(device)
      labels=labels.to(torch.long).to(device)
      logits,loss=model(input,labels)
      training_loss+=loss
      eval=evaluate(logits,labels)
      loss.backward()
      optim.step()
      acc+=eval['accuracy']
      recall+=eval['recall']
      prec+=eval['precision']
      if((st+1)%14==0):
        loss_curve.append(training_loss.detach().cpu()/(st+1))
        acc_curve.append(acc/(st+1))
        rec_curve.append(recall/(st+1))
        prec_curve.append(prec/(st+1))
    model.eval()
    with torch.no_grad():
      val_loss=0.0
      acc=0
      recall=0
      prec=0
      for st,(input,labels) in tqdm(enumerate(val_loader)):
        input=input.to(torch.float).to(device)
        labels=labels.to(torch.long).to(device)
        logits,loss=model(input,labels)
        val_loss+=loss
        eval=evaluate(logits,labels)
        acc+=eval['accuracy']
        recall+=eval['recall']
        prec+=eval['precision']
        if((st+1)%14==0):
          val_loss_curve.append(val_loss.detach().cpu()/(st+1))
          val_acc_curve.append(acc/(st+1))
          val_rec_curve.append(recall/(st+1))
          val_prec_curve.append(prec/(st+1))

  return {'Loss':[loss_curve,val_loss_curve],
          'Accuracy':[acc_curve,val_acc_curve],
          'Precision':[prec_curve,val_prec_curve],
          'Recall':[rec_curve,val_rec_curve]}

def eval(model,test_loader):
  with torch.no_grad():
      predictions=None
      ground_truth=None
      for st,(input,labels) in tqdm(enumerate(test_loader)):
        device=model.device
        input=input.to(torch.float).to(device)
        labels=labels.to(torch.long).to(device)
        logits,loss=model(input,labels)
        pred=np.array(logits.detach().cpu())
        pred=np.argmax(pred,axis=1)
        labs=np.array(labels.detach().cpu())
        pred=np.reshape(pred,(1,pred.shape[0]))
        labs=np.reshape(labs,(1,labs.shape[0]))
        if(predictions is None):
          predictions=pred
          ground_truth=labs
        else:
          predictions=np.concatenate([predictions,pred],axis=1)
          ground_truth=np.concatenate([ground_truth,labs],axis=1)
      return predictions,ground_truth


import warnings
warnings.filterwarnings('ignore')
from torch.optim import Adam
import matplotlib.pyplot as plt
def save_plots(plots,directory):
    for k in plots.keys():
        tr,val=plots[k]
        plt.plot([i for i in range(len(tr))],tr)
        plt.savefig('/'+directory+k+'_training.png')

        plt.plot([i for i in range(len(val))],val)
        plt.savefig('/'+directory+k+'_validation.png')



def train(mode,file_path,output_dir,model_path=None,epochs=5):
    if(mode=='train'):
        dataset = pd.read_csv(file_path)
        num_labels = len(pd.unique(dataset['labels']))
        id2label = dataset.set_index('ID')['Label'].to_dict()
        label2id = {}
        for i in id2label.keys():
            label2id[id2label[i]] = i

        train_dataset, val_dataset = train_test_split(dataset, test_size=0.3)
        val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.6)

        train_dataset = datasets(train_dataset['Path'], train_dataset['ID'])
        val_dataset = datasets(val_dataset['Path'], val_dataset['ID'])
        test_dataset = datasets(test_dataset['Path'], test_dataset['ID'])

        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=12, shuffle=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'label2id': label2id,
            'id2label': id2label,
            'name': 'ViT',
            'num_labels': len(id2label.keys())
        }
        ViT = model(config, 'vit_large_patch14_clip_224.openai_ft_in12k_in1k')
        if(model_path is not None):
            state_dict=torch.load(model_path)
            ViT.load_state_dict(state_dict)

        ViT.freeze_parameters()

        crit = Adam(ViT.parameters(), lr=0.001)
        losses = train_model(ViT.to(device), train_loader, val_loader, crit, epochs)

        save_plots(losses,output_dir)


        predictions, ground_truth = eval(ViT, test_loader)

        return evaluate(predictions,ground_truth)
    else:
        dataset = pd.read_csv(file_path)
        num_labels = len(pd.unique(dataset['labels']))
        id2label = dataset.set_index('ID')['Label'].to_dict()
        label2id = {}
        for i in id2label.keys():
            label2id[id2label[i]] = i

        dataset = datasets(dataset['Path'],dataset['ID'])
        loader = DataLoader(dataset, batch_size=12, shuffle=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'label2id': label2id,
            'id2label': id2label,
            'name': 'ViT',
            'num_labels': len(id2label.keys())
        }
        ViT = model(config, 'vit_large_patch14_clip_224.openai_ft_in12k_in1k')
        if(model_path is not None):
            ViT.load_state_dict(model_path)
        ViT.freeze_parameters()
        predictions, ground_truth = eval(ViT,loader)
        return evaluate(predictions, ground_truth)











