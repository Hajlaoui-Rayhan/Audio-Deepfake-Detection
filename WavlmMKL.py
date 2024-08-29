import os
import torch
from Data import CustData
from transformers import WavLMConfig, WavLMModel, AutoProcessor
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F




#mkl model
class MKLModel(nn.Module):

    def __init__(self, num_hidden_states):
        super().__init__()
        self.num_hidden_states = num_hidden_states
        # Learnable weights for each hidden state (kernel)
        self.kernel_weights = nn.Parameter(torch.ones(num_hidden_states, requires_grad=True))

    def forward(self, hidden_states):

        weighted_hidden_states = self.kernel_weights.unsqueeze(1) * hidden_states  # Shape: [batch_size, num_hidden_states, hidden_size]
        combined_output = torch.sum(weighted_hidden_states, dim=1)

        return combined_output

class CustomModel(nn.Module):
    def __init__(self, pretrained_model, num_hidden_states):
        super().__init__()
        #the WAVLM pretrained model
        self.pretrained_model = pretrained_model
        self.num_hidden_states=num_hidden_states
        #the mkl model
        self.mkl=MKLModel(num_hidden_states)
        # the classifier which is a feed forward
        self.classifier = nn.Linear(pretrained_model.config.hidden_size , 1)  # num hidden states and 1 is num calsses

    def forward(self, input_values):
        outputs = self.pretrained_model(input_values)
        last_hidden_states = outputs.last_hidden_state[:, -(self.num_hidden_states):, :]  # Get last n hidden states
        mkl_output= self.mkl(last_hidden_states)

        #flatten to pass it to the classifier
        pooled_output = torch.flatten(mkl_output, start_dim=1)  # Flatten for classifier input

        #classifier which is a simple fully connected layer
        logits = self.classifier(pooled_output)
        sigmoid = nn.Sigmoid()
        probs = sigmoid(logits)
        return probs
#get the dataset

train_folders = "D:/asvspoof5 data/flac_T/flac_T/"
train_target = "D:/asvspoof5 data/ASVspoof5.train.metadata.txt"
val_folders = "D:/asvspoof5 data/flac_D/flac_D/"
val_target = "D:/asvspoof5 data/ASVspoof5.dev.metadata.txt"
os.path.dirname(train_folders)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

audio_files = os.listdir(train_folders)
val_files = os.listdir(val_folders)
wav_input_16khz = torch.randn(1,10000)

#prepare the dataloder the data loader
train_dl = torch.utils.data.DataLoader(CustData(audio_files, train_folders, train_target), batch_size=32, shuffle=True)
val_dl = torch.utils.data.DataLoader(CustData(val_files, val_folders, val_target), batch_size=32, shuffle=True)
#intilize the model
pretrained_model= WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base")

num_hidden_states=5
model = CustomModel(pretrained_model, num_hidden_states)

#training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

weight_for_bonafide = 0.6

pos_weight = torch.tensor(weight_for_bonafide).to(device)
print(pos_weight )
optimizer = optim.AdamW(model.parameters(), lr=1e-6)
criterion = FocalLoss(alpha=0.75, gamma=1)

num_epochs = 3



best_val_loss = float('inf')
best_model_path = 'saved_models/best_wavlm-mkl2_model.pth'

# Lists to store loss values
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_val_loss = 0
    print(f'Training epoch: {epoch}')

    train_iter = iter(train_dl)
    val_iter = iter(val_dl)

    for _ in range(len(train_dl)):
        # Training batch
        try:
            batch_waveforms, batch_targets = next(train_iter)
        except StopIteration:
            break

        input_values = batch_waveforms.to(device)
        labels = batch_targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_values)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        print(f'Training Loss: {loss.item()}')

        total_loss += loss.item()

        # Validation batch
        model.eval()
        try:
            val_waveforms, val_targets = next(val_iter)
        except StopIteration:
            val_iter = iter(val_dl)
            val_waveforms, val_targets = next(val_iter)

        with torch.no_grad():
            val_input_values = val_waveforms.to(device)
            val_labels = val_targets.to(device)

            # Forward pass
            val_outputs = model(val_input_values)
            val_loss_batch = criterion(val_outputs, val_labels)

            total_val_loss += val_loss_batch.item()
            print(f'Validation Loss for batch: {val_loss_batch.item()}')

            # Check if this is the best model so far
            if val_loss_batch.item() < best_val_loss:
                best_val_loss = val_loss_batch.item()
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")


        model.train()  # Set the model back to training mode

    average_loss = total_loss / len(train_dl)
    average_val_loss = total_val_loss / len(train_dl)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {average_loss:.4f}, Average Validation Loss: {average_val_loss:.4f}")

    # Store the losses
    train_losses.append(average_loss)
    val_losses.append(average_val_loss)







# Save model to file
model_save_path = 'saved_models/wavlm-mkl2_model.pth'
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

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
