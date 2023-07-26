import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

import transformers
from transformers import Wav2Vec2Processor

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
    acc=accuracy_score(p.label_ids,preds)
    prec=precision_score(p.label_ids,preds,average='micro')
    rec=recall_score(p.label_ids,preds,average='micro')
    return {"accuracy": acc,
            'precision':prec,
            'recall':rec}
from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn
from transformers import Trainer

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
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
        device='cuda' if torch.cuda.is_available() else 'cpu'
        inputs = self._prepare_inputs(inputs)
        inputs=inputs.to(device)
        loss=self.compute_loss(model,inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.detach()
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import librosa
from transformers.file_utils import ModelOutput
from transformers import AutoModel,AutoConfig,AutoProcessor
from transformers import AutoFeatureExtractor
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path,sr=target_sampling_rate)
    #resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    #speech = resampler(speech_array).squeeze().numpy()
    return speech_array

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples['Path']]
    result = processor(speech_list, sampling_rate=target_sampling_rate)
    return result

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
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


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, ckpt,config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.wav2vec2 = AutoModel.from_pretrained(ckpt)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)
        #print(logits.shape)
        loss = None
        if labels is not None:
          loss_fn = torch.nn.CrossEntropyLoss()
          loss = loss_fn(logits,labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")
from transformers import TrainingArguments
global processor
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
        model_ckpt = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
        config = AutoConfig.from_pretrained(
            model_ckpt,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="wav2vec2_baseline",
        )
        processor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        target_sampling_rate = processor.sampling_rate

        train_dataset = train_dataset.to_dict(orient='list')
        train_dataset = Dataset.from_dict(train_dataset)

        val_dataset = val_dataset.to_dict(orient='list')
        val_dataset = Dataset.from_dict(val_dataset)

        test_dataset = test_dataset.to_dict(orient='list')
        test_dataset = Dataset.from_dict(test_dataset)

        train_dataset = train_dataset.map(
            preprocess_function,
            batch_size=32,
            batched=True,
            num_proc=4)

        val_dataset = val_dataset.map(
            preprocess_function,
            batch_size=32,
            batched=True,
            num_proc=4
        )

        test_dataset = test_dataset.map(
            preprocess_function,
            batch_size=32,
            batched=True,
            num_proc=4
        )

        train_dataset = train_dataset.rename_column('ID', 'labels')
        val_dataset = val_dataset.rename_column('ID', 'labels')
        test_dataset = test_dataset.rename_column('ID','labels')


        model = Wav2Vec2ForSpeechClassification(ckpt=model_ckpt, config=config)
        if(model_path is not None):
            state_dict=torch.load(model_path)
            model.load_state_dict(state_dict)

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        model.freeze_feature_extractor()

        training_args = TrainingArguments(
            output_dir="/"+output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
            learning_rate=1e-4,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            logging_steps=25
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = CTCTrainer(
            model=model.to(device),
            data_collator=data_collator,
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

        model_ckpt = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
        config = AutoConfig.from_pretrained(
            model_ckpt,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="wav2vec2_baseline",
        )
        processor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        target_sampling_rate = processor.sampling_rate

        dataset = dataset.to_dict(orient='list')
        dataset = dataset.rename_column('ID', 'labels')

        model = Wav2Vec2ForSpeechClassification(ckpt=model_ckpt, config=config)
        if(model_path is not None):
            model.load_state_dict(model_path)

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        training_args = TrainingArguments(
            output_dir="/" + output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
            learning_rate=1e-4,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            logging_steps=25
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = CTCTrainer(
            model=model.to(device),
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics
        )
        out = trainer.predict(dataset)
        result = compute_metrics(out)

        return result














