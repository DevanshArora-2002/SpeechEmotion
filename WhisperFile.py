import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoFeatureExtractor, WhisperForAudioClassification,AutoConfig
from datasets import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers.models.whisper.modeling_wav2vec2 import (
#     Wav2Vec2PreTrainedModel,
#     Wav2Vec2Model
# )


class ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self,ckpt,config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config=config
        self.model = WhisperForAudioClassification.from_pretrained(ckpt,config=config)
    def freeze_feature_extractor(self):
        for param in self.model.parameters():
          param.requires_grad=False
    def forward(
            self,
            input_values,
            labels=None,
    ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device=self.model.device
        outputs = self.model(
            input_values.to(device),
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
          loss_fn = torch.nn.CrossEntropyLoss()
          loss = loss_fn(logits,labels)
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
        )
from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn

from transformers import (
    Trainer,
    is_apex_available,
)

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class CTCTrainer(Trainer):
  def training_step(self, model: nn.Module, inputs) -> torch.Tensor:
      """
      Perform a training step on a batch of inputs.

      Subclass and override to inject custom behavior.

      Args:
          model (:obj:`nn.Module`):
              The model to train.
          inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
              The inputs and targets of the model.

              The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
              argument :obj:`labels`. Check your model's documentation for all accepted arguments.

      Return:
          :obj:`torch.Tensor`: The tensor with training loss on this batch.
      """
      model.train()
      inputs = self._prepare_inputs(inputs)
      #inputs=inputs.to(device)
      loss=self.compute_loss(model,inputs)
      if self.args.gradient_accumulation_steps > 1:
          loss = loss / self.args.gradient_accumulation_steps
      loss.backward()
      return loss.detach()
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
global preds_val
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds)
    preds=np.argmax(preds,axis=1)
    print(preds.shape)
    print(p.label_ids.shape)
    acc=accuracy_score(p.label_ids,preds)
    prec=precision_score(p.label_ids,preds,average='weighted')
    rec=recall_score(p.label_ids,preds,average='weighted')
    return {"accuracy": acc,
            'precision':prec,
            'recall':rec}

from sklearn.model_selection import train_test_split
def get_audio(path):
  sr=target_sampling_rate
  data,sr=librosa.load(path,sr=sr)
  return data
def preprocess(examples):
  lis=[get_audio(ex) for ex in examples['Path']]
  result=feature_extractor(lis,sampling_rate=target_sampling_rate,chunk_length=15)
  return result
global feature_extractor
global target_sampling_rate
def train(mode,file_path,output_dir,model_path=None,epochs=2):
    if(mode=='train'):
        dataset = pd.read_csv(file_path)
        num_labels = len(pd.unique(dataset['labels']))
        id2label = dataset.set_index('ID')['Label'].to_dict()
        label2id = {}
        for i in id2label.keys():
            label2id[id2label[i]] = i

        train_dataset, val_dataset = train_test_split(dataset, test_size=0.3)
        val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.6)
        ckpt = "openai/whisper-small"
        feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
        target_sampling_rate=feature_extractor.sampling_rate

        train_dataset = Dataset.from_dict(train_dataset.to_dict(orient='list'))
        val_dataset = Dataset.from_dict(val_dataset.to_dict(orient='list'))
        test_dataset = Dataset.from_dict(test_dataset.to_dict(orient='list'))

        train_dataset = train_dataset.map(
            preprocess,
            batch_size=32,
            batched=True,
            num_proc=True,
        )
        val_dataset = val_dataset.map(
            preprocess,
            batch_size=32,
            batched=True,
            num_proc=True,
        )
        test_dataset = test_dataset.map(
            preprocess,
            batch_size=32,
            batched=True,
            num_proc=True,
        )
        device='cuda' if torch.cuda.is_available() else 'cpu'
        train_dataset = train_dataset.rename_column('ID', 'labels')
        val_dataset = val_dataset.rename_column('ID', 'labels')
        test_dataset = test_dataset.rename_column('ID', 'labels')

        train_dataset = train_dataset.rename_column('input_features', 'input_values')
        val_dataset = val_dataset.rename_column('input_features', 'input_values')
        test_dataset = test_dataset.rename_column('input_features', 'input_values')

        train_dataset = train_dataset.remove_columns(['Path', 'Label'])
        val_dataset = val_dataset.remove_columns(['Path', 'Label'])
        test_dataset = test_dataset.remove_columns(['Path', 'Label'])

        num_classes = len(label2id.keys())


        config = AutoConfig.from_pretrained(ckpt, num_classes=num_classes,
                                            label2id=label2id,
                                            id2label=id2label, )

        model = Model(ckpt, config)
        if(model_path is not None):
            model.load_state_dict(model_path)
        # model.freeze_feature_extractor()
        from transformers import TrainingArguments

        training_args = TrainingArguments(
            logging_steps=50,
            output_dir="/"+output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
            learning_rate=1e-4,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
        )

        trainer = CTCTrainer(
            model=model.to(device),
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        out = trainer.predict(test_dataset)
        result = compute_metrics(out)
        return result
    else:
        dataset = pd.read_csv(file_path)
        num_labels = len(pd.unique(dataset['labels']))
        id2label = dataset.set_index('ID')['Label'].to_dict()
        label2id = {}
        for i in id2label.keys():
            label2id[id2label[i]] = i


        ckpt = "openai/whisper-small"
        feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
        target_sampling_rate = feature_extractor.sampling_rate

        dataset = Dataset.from_dict(dataset.to_dict(orient='list'))

        dataset = dataset.map(
            preprocess,
            batch_size=32,
            batched=True,
            num_proc=True,
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dataset = dataset.rename_column('ID', 'labels')
        dataset = dataset.rename_column('input_features', 'input_values')
        dataset = dataset.remove_columns(['Path', 'Label'])



        num_classes = len(label2id.keys())

        config = AutoConfig.from_pretrained(ckpt, num_classes=num_classes,
                                            label2id=label2id,
                                            id2label=id2label, )

        model = Model(ckpt, config)
        if (model_path is not None):
            state_dict=torch.load(model_path)
            model.load_state_dict(state_dict)
        # model.freeze_feature_extractor()
        from transformers import TrainingArguments

        training_args = TrainingArguments(
            logging_steps=50,
            output_dir="/" + output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
            learning_rate=1e-4,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
        )

        trainer = CTCTrainer(
            model=model.to(device),
            args=training_args,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        out = trainer.predict(dataset)
        result = compute_metrics(out)
        return result










