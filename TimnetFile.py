import keras.models
import tensorflow
import numpy as np
import pandas as pd
import librosa
import librosa
def get_feature(file_path: str, mfcc_len: int = 39, mean_signal_length: int = 100000):
  """
  file_path: Speech signal folder
  mfcc_len: MFCC coefficient length
  mean_signal_length: MFCC feature average length
  """
  signal, fs = librosa.load(file_path)
  s_len = len(signal)

  if s_len < mean_signal_length:
      pad_len = mean_signal_length - s_len
      pad_rem = pad_len % 2
      pad_len //= 2
      signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
  else:
      pad_len = s_len - mean_signal_length
      pad_len //= 2
      signal = signal[pad_len:pad_len + mean_signal_length]
  mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=39)
  mfcc = mfcc.T
  feature = mfcc
  return feature


import keras.backend as K
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D,add,GlobalAveragePooling1D
from keras.layers import BatchNormalization
from keras.activations import sigmoid

class Temporal_Aware_Block(tf.keras.layers.Layer):
  """
  Individual Temporal aware block consisting of a 1D convolution and
  attention model and returns the block
  """
  def __init__(self,s, i, activation, nb_filters, kernel_size, dropout_rate=0):
    super().__init__()
    self.conv1=Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')
    self.conv2=Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal')
    self.bn1=BatchNormalization(trainable=True,axis=-1)
    self.bn2=BatchNormalization(trainable=True,axis=-1)
    self.ac1=Activation(activation)
    self.ac2=Activation(activation)
    self.d1=SpatialDropout1D(dropout_rate)
    self.d2=SpatialDropout1D(dropout_rate)
    self.nb_filters=nb_filters
  
  def compute_output_shape(self,input_shape):
    return (1,self.nb_filters)
  def call(self,x):
    original_x = x
    #1.1
    conv_1_1 = self.conv1(x)
    conv_1_1 = self.bn1(conv_1_1)
    conv_1_1 =  self.ac1(conv_1_1)
    output_1_1 =  self.d1(conv_1_1)
    # 2.1
    conv_2_1 = self.conv2(output_1_1)
    conv_2_1 = self.bn2(conv_2_1)
    conv_2_1 = self.ac2(conv_2_1)
    output_2_1 =  self.d2(conv_2_1)

    if original_x.shape[-1] != output_2_1.shape[-1]:
        original_x = Conv1D(filters=self.nb_filters, kernel_size=1, padding='same')(original_x)

    output_2_1 = Lambda(sigmoid)(output_2_1)
    F_x = Lambda(lambda x: tf.multiply(x[0], x[1]))([original_x, output_2_1])
    return F_x

class WeightLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x,[0,2,1])
        x = K.dot(tempx,self.kernel)
        x = tf.squeeze(x,axis=-1)
        return  x

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

class TIMNET(tf.keras.layers.Layer):
    """
    Main TIMNET model consisting of multiple Temporal Aware Blocks
    based on filters, kernels stacks and dialations
    """
    def __init__(self,num_labels,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=8,
                 activation = "relu",
                 dropout_rate=0.1,
                 return_sequences=True):
      super().__init__()
      self.flatten = tf.keras.layers.Flatten()
      self.return_sequences = return_sequences
      self.activation = activation
      self.dropout_rate = dropout_rate
      self.dialations = dilations
      self.nb_stacks = nb_stacks
      self.kernel_size = kernel_size
      self.nb_filters = nb_filters

      self.supports_masking = True
      self.mask_value=0.
      self.num_labels=num_labels

      self.conv1=Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')
      self.conv2=Conv1D(filters=self.nb_filters,kernel_size=1, dilation_rate=1, padding='causal')
      self.list_for=[]
      self.list_back=[]
      self.d1=tf.keras.layers.Dense(units=512,activation='relu')
      self.d2=tf.keras.layers.Dense(units=self.num_labels,activation='softmax')
      for s in range(self.nb_stacks):
        self.list_for.append([])
        self.list_back.append([])
        for i in [2 ** i for i in range(self.dialations)]:
            self.list_for[s].append(Temporal_Aware_Block(s, i, self.activation,
                                                    self.nb_filters,
                                                    self.kernel_size,
                                                    self.dropout_rate))
            self.list_back[s].append(Temporal_Aware_Block(s, i, self.activation,
                                                    self.nb_filters,
                                                    self.kernel_size,
                                                    self.dropout_rate))

      self.weight_layer=WeightLayer()

    def call(self, inputs, mask=None):
        forward = inputs
        backward = K.reverse(inputs,axes=1)

        forward_convd = self.conv1(forward)
        backward_convd = self.conv2(backward)

        final_skip_connection = []

        skip_out_forward = forward_convd
        skip_out_backward = backward_convd

        for s in range(self.nb_stacks):
          for i in range(self.dialations):
            skip_out_forward = self.list_for[s][i](skip_out_forward)
            skip_out_backward = self.list_back[s][i](skip_out_backward)

            temp_skip = add([skip_out_forward, skip_out_backward])
            temp_skip=GlobalAveragePooling1D()(temp_skip)
            temp_skip=tf.expand_dims(temp_skip, axis=1)
            final_skip_connection.append(temp_skip)

        output_2 = final_skip_connection[0]
        for i,item in enumerate(final_skip_connection):
          if i==0:
              continue
          output_2 = K.concatenate([output_2,item],axis=-2)
        x = output_2
        x=self.weight_layer(x)
        x=self.d1(x)
        x=self.d2(x)
        return x

from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score
def evaluate(pred,labels):
  p1=np.array(pred)
  l1=np.array(labels)
  p1=np.argmax(p1,axis=1)
  return accuracy_score(p1,l1),precision_score(p1,l1,average='weighted'),recall_score(p1,l1,average='weighted')
def train_model(train_dataset,val_dataset,epochs,loss_fn,optim,model):
  train_loss=[]
  train_acc=[]
  train_prec=[]
  train_rec=[]
  val_loss=[]
  val_acc=[]
  val_prec=[]
  val_rec=[]
  for ep in range(epochs):
    batch_size = 12
    train_iter = iter(train_dataset)
    tr_loss=0
    acc=0
    prec=0
    rec=0
    for i,batch in tqdm(enumerate(train_iter)):
      inputs, labels = batch
      with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        predictions = model(inputs)
        loss = loss_fn(from_logits=False)(labels, predictions)
      tr_loss+=loss.numpy()
      a,p,r=evaluate(predictions,labels)
      acc+=a
      prec+=p
      rec+=r
      gradients = tape.gradient(loss, model.trainable_variables)
      optim.apply_gradients(zip(gradients, model.trainable_variables))
      if(i+1)%20==0:
        train_loss.append(tr_loss/(i+1))
        train_acc.append(acc/(i+1))
        train_prec.append(prec/(i+1))
        train_rec.append(rec/(i+1))
    v_loss=0
    acc=0
    prec=0
    rec=0
    val_iter=iter(val_dataset)
    for i,batch in tqdm(enumerate(val_iter)):
      inputs, labels = batch
      predictions=model(inputs)
      loss = loss_fn(from_logits=False)(labels, predictions)
      v_loss+=loss.numpy()
      a,p,r=evaluate(predictions,labels)
      acc+=a
      prec+=p
      rec+=r
      if(i+1)%14==0:
        val_loss.append(v_loss/(i+1))
        val_acc.append(acc/(i+1))
        val_prec.append(prec/(i+1))
        val_rec.append(rec/(i+1))
  return {'loss':[train_loss,val_loss],
          'acc':[train_acc,val_acc],
          'prec':[train_prec,val_prec],
          'rec':[train_rec,val_rec]}

def train_test_split(input_x,target_y,split=0.5):
    # Selective train-test-split based on split size 
    #returns training,test
    no_samples=input_x.shape[0]
    no_train_samples=int(no_samples*0.5)

    train_x=input_x[:no_train_samples,:,:]
    train_y=target_y[:no_train_samples,:]

    val_x=input_x[no_train_samples:,:,:]
    val_y=target_y[no_train_samples:,:,:]

    return train_x,val_x,train_y,val_y
def predict(test_x,labels,loss_fn,model):
  pred=model(test_x)
  loss = loss_fn(from_logits=False)(labels, pred)
  return loss,np.argmax(np.array(pred),axis=1)

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import matplotlib.pyplot as plt
def save_plots(plots,directory):
    for k in plots.keys():
        tr,val=plots[k]
        plt.plot([i for i in range(len(tr))],tr)
        plt.savefig('/'+directory+k+'_training.png')

        plt.plot([i for i in range(len(val))],val)
        plt.savefig('/'+directory+k+'_validation.png')
def train(mode,file_path,output_dir,model_path=None,epochs=2):
    if(mode=='train'):
        df = pd.read_csv(file_path)
        num_labels = len(pd.unique(df['labels']))
        id2label = df.set_index('ID')['Label'].to_dict()
        label2id = {}
        for i in id2label.keys():
            label2id[id2label[i]] = i

        input = None
        y = []

        for i in tqdm(range(len(df['Path']))):
            if (input is None):
                f = get_feature(df['Path'].iloc[i])
                input = np.reshape(f, (1, f.shape[0], f.shape[1]))
            else:
                f = get_feature(df['Path'].iloc[i])
                f = np.reshape(f, (1, f.shape[0], f.shape[1]))
                input = np.concatenate([input, f], axis=0)
            y.append(df['ID'].iloc[i])

        y = []
        for i in range(len(df['ID'])):
            y.append(df['ID'].iloc[i])

        train_x,val_x,train_y,val_y=train_test_split(input,y,0.8)
        val_x,test_x,val_y,test_y=train_test_split(val_x,val_y,0.5)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

        batch_size = 12
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        if(model_path is not None):
            model=keras.models.load_model(model_path)
        else:
            model = tf.keras.Sequential()
            model.add(TIMNET(num_labels=len(label2id.keys()), nb_stacks=10))
            input = np.zeros((12, 196, 39))
            out = model(input)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=[tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall()])
        print(model.summary())
        loss_fn = tf.losses.SparseCategoricalCrossentropy
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        losses = train_model(train_dataset, val_dataset, epochs, loss_fn, optimizer, model)
        save_plots(losses,output_dir)

        loss, pred_labels = predict(test_x, test_y, loss_fn, model)
        a,p,r=evaluate(pred_labels,test_y)
        dict={'Accuracy':a,
              "Precision":p,
              "Recall":r}
        return dict

    else:
        df = pd.read_csv(file_path)
        num_labels = len(pd.unique(df['labels']))
        id2label = df.set_index('ID')['Label'].to_dict()
        label2id = {}
        for i in id2label.keys():
            label2id[id2label[i]] = i

        input = None
        y = []

        for i in tqdm(range(len(df['Path']))):
            if (input is None):
                f = get_feature(df['Path'].iloc[i])
                input = np.reshape(f, (1, f.shape[0], f.shape[1]))
            else:
                f = get_feature(df['Path'].iloc[i])
                f = np.reshape(f, (1, f.shape[0], f.shape[1]))
                input = np.concatenate([input, f], axis=0)
            y.append(df['ID'].iloc[i])

        y = []
        for i in range(len(df['ID'])):
            y.append(df['ID'].iloc[i])

        model = tf.keras.Sequential()
        model.add(TIMNET(num_labels=len(label2id.keys()), nb_stacks=10))
        input = np.zeros((12, 196, 39))
        out = model(input)
        print(model.summary())
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])

        loss_fn = tf.losses.SparseCategoricalCrossentropy
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        loss, pred_labels = predict(input,y, loss_fn, model)
        a, p, r = evaluate(pred_labels,y)
        dict = {'Accuracy': a,
                "Precision": p,
                "Recall": r}
        return dict









