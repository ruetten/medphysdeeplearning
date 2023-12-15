import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import csv
import loader


class CSVDataset(Dataset):
    def __init__(self, filepath='predictions_5.csv', num_features=5, num_channels=1, num_classes=2):
        self.filepath = filepath
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_samples, self.data, self.labels = self.generate_data()

    def generate_data(self):
        data = []
        labels = []
        PTIDs = []
        imgIDs = []

        csv_file_path = self.filepath

        with open(csv_file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                label = row['Class']
                features = torch.tensor([float(item) for item in row['Predictions'].strip('[]').split()]).reshape(1, num_features-1)
                ptid = row['PTIDs']

                data.append(features)
                labels.append(torch.tensor(int(label)).view(1))
                PTIDs.append(ptid)
                imgIDs.append(row['ImageIDs'])

        # Create a dictionary to store unique Features for each PTID
        ptid_features_dict = {}
        ptid_labels_dict = {}
        ptid_imgID_dict = {}

        # Iterate through the data and populate the dictionary
        for ptid, feature, label, imgID in zip(PTIDs, data, labels, imgIDs):
        #for ptid, feature, label in zip(PTIDs, data, labels):
            if ptid in ptid_features_dict:
                ptid_features_dict[ptid].append(feature)
                ptid_imgID_dict[ptid].append(imgID)
            else:
                ptid_features_dict[ptid] = [feature]
                ptid_labels_dict[ptid] = label
                ptid_imgID_dict[ptid] = [imgID]

        data = []
        labels = []
        the_fields = loader.getData()
        for key, value in ptid_features_dict.items():
            imgIDs = ptid_imgID_dict[key]

            # Entry point code here just for debugging
            the_result, imgIDs_time = loader.parseTimestamps(key, the_fields)
            #print("VIT1", the_result, imgIDs_time)
            #print(imgIDs)
            time_embed = []
            for imgID in imgIDs:
                try:
                    idx = imgIDs_time[0].index(imgID)
                    time_embed.append(the_result[0][idx])
                except ValueError:
                    time_embed.append(-1)
            while len(time_embed) < 3:
                time_embed.append(-1)
            print(time_embed)
            
            result_tensor = torch.concatenate(value, axis=1)

            # Check if the result tensor is less than 15 columns
            if result_tensor.shape[1] < (num_features-1)*3:
                # Calculate the number of columns to pad
                padding_size = (num_features-1)*3 - result_tensor.shape[1]

                # Create a padding array with -1 and concatenate it to the result_tensor
                padding_array = torch.full((1, padding_size), -1)
                result_tensor = torch.concatenate([result_tensor, padding_array], axis=1)

            #result_tensor = torch.concatenate([result_tensor, torch.tensor(time_embed).unsqueeze(0)], axis=1)

            final_result_tensor = []

            print(result_tensor)
            i = 0
            for time in time_embed:
                for v in range(num_features-1):
                    final_result_tensor.append(result_tensor[0][i+v])
                i = i + num_features-1
                final_result_tensor.append(time)
            final_result_tensor = torch.tensor(final_result_tensor).unsqueeze(0)

            print(final_result_tensor)
            data.append(final_result_tensor)
            labels.append(ptid_labels_dict[key])

        return len(ptid_labels_dict), torch.stack(data).unsqueeze(0), torch.tensor(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[0][idx], self.labels[idx]

# Dummy Dataset class
class DummyDataset(Dataset):
    def __init__(self, num_samples, num_features, num_channels, num_classes=2):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.data, self.labels = self.generate_data()
    
    def generate_data(self):
        data = []
        labels = []

        for _ in range(self.num_samples):
            # Randomly choose a class
            class_label = torch.randint(0, self.num_classes, (1,)).item()

            # Generate data based on the class label
            if class_label == 0:
                # Data for class 0 from a normal distribution with mean 0 and standard deviation 1
                sample = torch.randn((self.num_channels, self.num_features))
                sample = torch.clamp(sample, -1, 0)  # Normalize to the range [-1, 1]
            else:
                # Data for class 1 from a normal distribution with mean 5 and standard deviation 1
                sample = torch.randn((self.num_channels, self.num_features))
                sample = torch.clamp(sample, 0, 1)  # Normalize to the range [-1, 1]

            print(sample, type(sample), sample.shape)
            data.append(sample)
            labels.append(class_label)

        return torch.stack(data), torch.tensor(labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

if __name__ == '__main__':
    # Create dummy dataset
    num_channels = 1
    num_features = 6
    num_classes = 2

    #dummy_dataset = DummyDataset(num_samples=1000, num_features=num_features, num_channels=num_channels, num_classes=num_classes)
    dummy_dataset = CSVDataset(filepath='predictions_5.csv', num_features=num_features)

    #print(dummy_dataset.data)
    #print(len(dummy_dataset.data[0][0]))
    #print(len(dummy_dataset.labels))

    # Split dataset into train and validation sets
    num_samples = len(dummy_dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    #print(len(dummy_dataset.labels), train_size, val_size, train_size + val_size)

    train_dataset, val_dataset = torch.utils.data.random_split(dummy_dataset, [train_size, val_size])
    # Define the K-fold cross-validation
    #kf = KFold(n_splits=5, shuffle=True, random_state=42)

    #for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
        #print(f"Fold {fold + 1}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Check a batch from the train loader
    #for inputs, labels in train_loader:
        #print("Input Shape:", inputs.shape)
        #print("Input:", inputs)
        #print("Labels:", labels)
        #break
    
    model = ViT(
        seq_len = 18, #(num_features+1)*3,
        patch_size = 6, #num_features+1,
        num_classes = 2,
        channels = 1,
        dim = 32,
        depth = 6,
        heads = 8,
        mlp_dim = 128,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    #time_series = torch.randn(4, 1, 15)
    #print(time_series)
    #logits = v(time_series) # (4, 2)
    #print(logits)

    # Choose loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:  # Assuming you have a train_loader
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:  # Assuming you have a val_loader
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), 'vit_time_series.pth')

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:  # Assuming you have a val_loader
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to NumPy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate accuracy or other metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Calculate sensitivity and specificity
    true_positive = conf_matrix[1, 1]
    false_negative = conf_matrix[1, 0]
    true_negative = conf_matrix[0, 0]
    false_positive = conf_matrix[0, 1]
    
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_predictions) 
    
    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC: {auc:.4f}")

    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
