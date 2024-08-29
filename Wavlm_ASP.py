"""
__author__ = Yi Zhu
__email__ = Yi.Zhu@inrs.ca

This script is used to define baselines deepfake detection models with SSL frontends. The front and backend modules are defined separately but should always
be used together in the following sequence: frontend -> backend_clf.
"""
import speechbrain
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from transformers import WavLMModel
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import AutoProcessor, Wav2Vec2Model, HubertModel, Data2VecAudioModel
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
import torchaudio
import pandas
import os
from torch.utils.data import DataLoader
print(speechbrain.__version__)
device = "cuda"

train_folders = "D:/asvspoof5 data/flac_T/flac_T/"
train_target = "D:/asvspoof5 data/ASVspoof5.train.metadata.txt"
os.path.dirname(train_folders)
val_folders = "D:/asvspoof5 data/flac_D/flac_D/"
val_target = "D:/asvspoof5 data/ASVspoof5.dev.metadata.txt"
audio_files = os.listdir(train_folders)

wav_input_16khz = torch.randn(10000)

class SSL_frontend(nn.Module):
    def __init__(
        self,
        encoder_choice:list=['wav2vec','wavlm','hubert'],
        specify_layer:dict={'wav2vec':None,'wavlm':None,'hubert':None},
        freeze_encoder: bool = True,
        freeze_CNN: bool = True,
        device='cuda',
        *args,
        **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        # Load backbones (one or multiple): Wav2vec-XLSR, WavLM-Large, or HuBert-Large
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.freeze_CNN = freeze_CNN
        self.processor = Wav2Vec2FeatureExtractor() # front-end audio processor remains the same
        self.encoder_choice = encoder_choice

        if 'data2vec-base' in encoder_choice:
            self.d2v = self.init_sfm('data2vec-base')
            self._lock_encoder(self.d2v)
        if 'wav2vec' in encoder_choice:
            self.w2v = self.init_sfm('wav2vec')
            self._lock_encoder(self.w2v)
        if 'wav2vec-ser' in encoder_choice:
            self.w2v = self.init_sfm('wav2vec-ser')
            self._lock_encoder(self.w2v)
        if  'wav2vec-asr' in encoder_choice:
            self.w2v = self.init_sfm('wav2vec-asr')
            self._lock_encoder(self.w2v)
        if 'wav2vec-large' in encoder_choice:
            self.w2v = self.init_sfm('wav2vec-large')
            self._lock_encoder(self.w2v)
        if 'wavlm' in encoder_choice:
            self.wlm = self.init_sfm('wavlm')
            self._lock_encoder(self.wlm)
        if 'wavlm-base' in encoder_choice:
            self.wlm = self.init_sfm('wavlm-base')
            self._lock_encoder(self.wlm)
        if 'hubert' in encoder_choice:
            self.hub = self.init_sfm('hubert')
            self._lock_encoder(self.hub)
        if 'hubert-base' in encoder_choice:
            self.hub = self.init_sfm('hubert-base')
            self._lock_encoder(self.hub)            

        self.specify_layer = specify_layer
        self._init_num_layer()
    
    def _init_num_layer(self):
        self.total_num_layer = 0
        for _,l in enumerate(self.specify_layer):
            if self.specify_layer[l] is None: self.total_num_layer += 24
            elif isinstance(self.specify_layer[l], int): self.total_num_layer += 1
            elif isinstance(self.specify_layer[l], list): self.total_num_layer += len(self.specify_layer[l])

    def _lock_encoder(self,m):
        for param in m.parameters():
            param.requires_grad = not self.freeze_encoder
        if self.freeze_CNN:
            m.feature_extractor.eval()
            for param in m.feature_extractor.parameters():
                param.requires_grad = False

    def init_sfm(self,encoder_choice):
        ssl_encoder_source = {
             'wav2vec': 'facebook/wav2vec2-large-xlsr-53',
             'wav2vec-large': 'facebook/wav2vec2-large-960h',
             'wavlm-bp': 'microsoft/wavlm-base-plus',
             'wavlm':'microsoft/wavlm-large',
             'wavlm-base': 'microsoft/wavlm-base',
             'hubert':'facebook/hubert-large-ll60k',
             'hubert-base':'facebook/hubert-base-ls960',
             'wav2vec-asr': 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
             'wav2vec-ser': 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
             'data2vec-base': 'facebook/data2vec-audio-base-960h',
        }
        assert encoder_choice in list(ssl_encoder_source.keys()), \
            f"Currently supported encoder types include {list(ssl_encoder_source.keys())}"
        if ('wavlm' in encoder_choice) or ('wavlm-base' in encoder_choice):
            feature_extractor = WavLMModel.from_pretrained(ssl_encoder_source[encoder_choice])
        elif ('wav2vec' in encoder_choice) or ('wav2vec-large' in encoder_choice):
            feature_extractor = Wav2Vec2Model.from_pretrained(ssl_encoder_source[encoder_choice])
        elif ('hubert' in encoder_choice) or ('hubert-base' in encoder_choice):
            feature_extractor = HubertModel.from_pretrained(ssl_encoder_source[encoder_choice])
        elif ('data2vec-base' in encoder_choice):
            feature_extractor = Data2VecAudioModel.from_pretrained(ssl_encoder_source[encoder_choice])
        return feature_extractor

    def extract_feats(self,x,processor,encoder,layer):
        input_values = processor(x, sampling_rate=16000, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)
        features = encoder(input_values, output_hidden_states=True)
        if layer is None:
            layer = slice(0,None,1)
        features = torch.stack(features.hidden_states[1:],axis=0) # (layer=25, batch, time, features) *first layer is discarded
        features = features[layer,::]
        # border condition check
        if features.ndim < 4: 
            features = features.unsqueeze(0) # (layer=1, batch, time, features)
        return features

    def _extract_feats(self,x):
        all_emds = []
        if hasattr(self,'w2v'): all_emds.append(self.extract_feats(x,self.processor,self.w2v,layer=self.specify_layer['wav2vec']))
        if hasattr(self,'wlm'): all_emds.append(self.extract_feats(x,self.processor,self.wlm,layer=self.specify_layer['wavlm']))
        if hasattr(self,'hub'): all_emds.append(self.extract_feats(x,self.processor,self.hub,layer=self.specify_layer['hubert']))
        if hasattr(self,'d2v'): all_emds.append(self.extract_feats(x,self.processor,self.d2v,layer=self.specify_layer['data2vec']))
        all_emds = torch.cat(all_emds,dim=0)
        return all_emds

    def forward(self, x):
        # frontend
        features = self._extract_feats(x)
        return features

class backend_clf(nn.Module):
    def __init__(
        self,
        total_num_layer: int,
        pooling: str = "atn",
        num_ssl_features: int = 1024,
        num_fc_neurons:int = -1,
        num_classes: int = 1,
        dp = 0.25,
        device='cuda',
        *args,
        **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        self.total_num_layer = total_num_layer
        self.weights_stack = nn.Parameter(torch.ones(self.total_num_layer))
        
        num_fc_features = 2*num_ssl_features if pooling == 'atn' else num_ssl_features
        if num_fc_neurons == -1: num_fc_neurons = num_fc_features 
    
        self.fc = nn.Sequential(
            nn.Linear(num_fc_features, num_fc_neurons),
            nn.Dropout(p=dp),
            nn.LeakyReLU(0.1),
            nn.Linear(num_fc_features, num_classes)
        )

        # Pooling layers
        if pooling == "avg":
            self.pooling = lambda x: F.adaptive_avg_pool1d(x, 1)
        elif pooling == 'atn':
            self.pooling = AttentiveStatisticsPooling(
                num_ssl_features,
                attention_channels=num_ssl_features,
                global_context=True
            )

    def weighted_sum(self, features):
        """
        Returns a weighted sum of outputs from all layers.
        """
        layer_num = features.shape[0]
        _, *origin_shape = features.shape
        stacked_feature = features.view(layer_num, -1)
        norm_weights = F.softmax(self.weights_stack, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)
        return weighted_feature
    
    def forward(self,features):
        # backend modules:
        features = self.weighted_sum(features)
        features = features.permute(0, 2, 1) # (batch, time, features )=> (batch, features, time)
        features = self.pooling(features).squeeze(-1)
        output = self.fc(features)
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)





def resize_tensor(tensor, target_size=10000):
    # Flatten the tensor
    tensor = tensor.flatten()
    current_size = tensor.size(0)

    if current_size < target_size:
        # Pad with zeros if the tensor is smaller than the target size
        padding_size = target_size - current_size
        tensor = F.pad(tensor, (0, padding_size), 'constant', 0)
    elif current_size > target_size:
        # Trim the tensor if it is larger than the target size
        tensor = tensor[:target_size]

    # Reshape to (1, target_size)
    tensor = tensor.view(target_size)
    return tensor


class CustData(torch.utils.data.Dataset) :
    def __init__(self,audio_files, train_folders, train_target):
        self.audio_files = audio_files
        self.train_folders = train_folders
        self.train_target = train_target
    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, idx):
        audio_name = self.audio_files[idx]

        audio_sample, sr = torchaudio.load(self.train_folders + audio_name)
        audio_sample=audio_sample.to(device)
        audio_sample = resize_tensor(audio_sample)
        audio_sample = torch.nn.functional.layer_norm(audio_sample, wav_input_16khz.shape)

        pd = pandas.read_csv(self.train_target, sep=' ', header=None)




        target = pd.loc[pd[1]+'.flac' == audio_name][5]

        if (target == "spoof").all():
            target = 0
        else:
            target = 1
        return audio_sample, target
class CombinedModel(nn.Module):
    def __init__(self, frontend, backend):
        super(CombinedModel, self).__init__()
        self.frontend = frontend
        self.backend = backend

    def forward(self, x):
        features = self.frontend(x)
        outputs = self.backend(features)
        return outputs

val_files = os.listdir(val_folders)

train_dl = torch.utils.data.DataLoader(CustData(audio_files, train_folders, train_target),shuffle=True,batch_size=24)
val_dl = torch.utils.data.DataLoader(CustData(val_files, val_folders, val_target), batch_size=24, shuffle=True)
frontend = SSL_frontend(encoder_choice=['wavlm'], specify_layer={'wavlm': None}, device='cuda')
backend = backend_clf(total_num_layer=24, pooling='atn', num_ssl_features=1024, num_classes=2, device='cuda')  # Adjust num_classes based on your task
combined_model = CombinedModel(frontend, backend).to('cuda')
frontend.to('cuda')
backend.to('cuda')


optimizer = torch.optim.AdamW(combined_model.parameters(), lr=1e-4)
weight_for_bonafide = 0.7
weights = torch.tensor([0.2, 0.8], dtype=torch.float32)
pos_weight = torch.tensor(weight_for_bonafide).to(device)
if torch.cuda.is_available():
    weights = weights.cuda()
criterion = nn.CrossEntropyLoss(weight=weights)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
num_epochs = 3  # Number of epochs
train_losses = []
val_losses = []
best_model_path = 'saved_models/'
model_path = 'saved_models/'
gradient_clip_value = 1.0
best_val_loss = float('inf')
for epoch in range(num_epochs):
    combined_model.train()
    total_loss = 0
    total_val_loss = 0
    print(f'Training epoch: {epoch+1}/{num_epochs}')

    train_iter = iter(train_dl)
    val_iter = iter(val_dl)

    for _ in range(len(train_dl)):
        # Training batch
        try:
            audio, labels = next(train_iter)
        except StopIteration:
            break

        input_values = audio.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()

        # Forward pass

        outputs = combined_model(input_values)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(combined_model.parameters(), gradient_clip_value)

        optimizer.step()
        scheduler.step()
        print(f'Training Loss: {loss.item()}')

        total_loss += loss.item()

        # Validation batch
        frontend.eval()
        backend.eval()
        try:
            val_audio, val_labels = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dl)
            val_audio, val_labels = next(val_iter)

        with torch.no_grad():
            val_input_values = val_audio.to('cuda')
            val_labels = val_labels.to('cuda')

            # Forward pass
            val_outputs = combined_model(val_input_values)
            val_loss_batch = criterion(val_outputs, val_labels)

            total_val_loss += val_loss_batch.item()
            print(f'Validation Loss for batch: {val_loss_batch.item()}')

            # Check if this is the best model so far
            if val_loss_batch.item() < best_val_loss:
                best_val_loss = val_loss_batch.item()
                torch.save(combined_model.state_dict(), best_model_path + '_combined.pth')
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        frontend.train()  # Set the frontend back to training mode
        backend.train()   # Set the backend back to training mode

    average_loss = total_loss / len(train_dl)
    average_val_loss = total_val_loss / len(val_dl)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {average_loss:.4f}, Average Validation Loss: {average_val_loss:.4f}")

    # Store the losses
    train_losses.append(average_loss)
    val_losses.append(average_val_loss)


 # Store the losses
    train_losses.append(average_loss)
    val_losses.append(average_val_loss)

torch.save(combined_model.state_dict(), model_path + '_combined_final.pth')

print("Training complete!")
# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
