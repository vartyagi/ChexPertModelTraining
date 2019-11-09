import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils, datasets
from sklearn.metrics import roc_auc_score
import time
import copy
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

print(torch.cuda.current_device())

print(torch.cuda.is_available())

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 5

# Batch size for training (change depending on how much memory you have)
batch_size = 48

# Number of epochs to train for
num_epochs = 2

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 320

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        input_size = 320

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

print(model_ft)

# Read training data
df = pd.read_csv('TrainingData/frontal_images_distance_score5.csv').iloc[:,1:]
df.replace(-1,0, inplace=True)
df['NewPath'] = 'TrainingData/' + df['NewPath']
df = df[df['DiscardFlag'] == 0]

df = df[['NewPath', 'Cardiomegaly', 'Edema', 'Atelectasis',
         'Pleural Effusion', 'Consolidation']]

#Read test data
dir_ = 'CheXpert-v1.0-small/'
df_valid = pd.read_csv('TrainingData/valid.csv')
a = df_valid['Path'].str.split("/",expand=True)
df_valid['patient']=a[2].str.replace('patient','').astype(int)
df_valid['study'] = a[3].str.replace('study','').astype(int)
df_valid['view'] = a[4].str.split('_', expand=True)[0].str.replace('view','').astype(int)
df_valid = df_valid[df_valid['Frontal/Lateral'] == 'Frontal']
df_valid['NewPath'] = df_valid['Path'].str.replace(dir_, 'TrainingData/')
df_valid = df_valid[['NewPath', 'Cardiomegaly', 'Edema', 'Atelectasis',
         'Pleural Effusion', 'Consolidation']]

train, valid = train_test_split(df, test_size=0.05, random_state=42)
count_labels_df = train.iloc[:,1:9].apply(lambda x: 100 * x.value_counts()/ x.count())

ratio = count_labels_df.iloc[0]/count_labels_df.iloc[1]

weights = torch.from_numpy(ratio.values).float()


class XrayDataset(Dataset):

    def __init__(self, df, transform=None):
        self.xray_frame = df
        self.transform = transform
        self.y_train = np.array(self.xray_frame.iloc[:, 1:9]).astype(float)

    def __len__(self):
        return len(self.xray_frame)

    def __getitem__(self, idx):
        img_name = self.xray_frame['NewPath'].iloc[idx]
        image = Image.open(img_name)
        target_labels = torch.from_numpy(self.y_train[idx])
        if self.transform:
            image = self.transform(image)
        sample = image, target_labels
        return sample


class XrayDataset_test(Dataset):
    def __init__(self, df, transform=None):
        self.xray_frame = df
        self.transform = transform
        self.y_train = np.array(self.xray_frame.iloc[:, 1:9]).astype(float)

    def __len__(self):
        return len(self.xray_frame)

    def __getitem__(self, idx):
        img_name = self.xray_frame['NewPath'].iloc[idx]
        image = Image.open(img_name)
        target_labels = torch.from_numpy(self.y_train[idx])

        if self.transform:
            image = self.transform(image)
        sample = image, target_labels
        return sample

transform_train = transforms.Compose([
    transforms.Resize(320),
    transforms.ToTensor(),
    transforms.Normalize([0.5247], [0.2769])
    ])

transform_valid = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize([0.5247], [0.2769])
    ])


data = {}
data['train'] = XrayDataset(train, transform=transform_train)
data['valid'] = XrayDataset_test(valid, transform=transform_train)
data['test'] = XrayDataset_test(df_valid, transform=transform_valid)

dataloaders_dict = {x: torch.utils.data.DataLoader(data[x], batch_size=batch_size, shuffle=False, num_workers=4)
                    for x in ['train', 'valid', 'test']}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
#optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = torch.optim.Adam(params_to_update, lr=1e-4)

labels_list = ['Cardiomegaly', 'Edema', 'Atelectasis', 'Pleural Effusion', 'Consolidation']


def auroc_metric(outputs, labels):
    auc_scores = [roc_auc_score(labels[:,i], outputs[:,i]) for i in range(len(labels_list))]
    return np.sum(auc_scores)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    history = {'train': [], 'valid': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999
    best_performance = 0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            history[phase].append([])  # append empty history for current epoch
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            full_labels = []
            full_preds = []

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.float()
                labels = labels.to(device)
                full_labels.append(np.array(labels.cpu()))
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    outputs = torch.sigmoid(outputs)
                    full_preds.append(np.array(outputs.cpu().detach().numpy()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if batch_idx % 10 == 0:
                    history[phase][epoch].append({'Loss' : loss.data.cpu()})
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                          .format(epoch,
                                  batch_idx * len(data),
                                  len(dataloaders[phase].dataset),
                                  100. * batch_idx / len(dataloaders[phase]),
                                  loss.data))

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'valid':
                full_labels = np.concatenate(full_labels)
                full_preds = np.concatenate(full_preds)
                performance = auroc_metric(full_preds, full_labels)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            history[phase][epoch].append({'EpochLoss': epoch_loss})

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'valid':
                print('{} SUM(AUCROC): {:.4f}'.format(phase, performance))
                history[phase][epoch].append({'AUCROC': performance})
            # deep copy the model
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss

            if phase == 'valid' and performance > best_performance:
                best_performance = performance
                print("new model copied")
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val SUM(AUCROC): {:4f}'.format(best_performance))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.float().to(device))


model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                             is_inception=(model_name=="inception"))
f = open('history.txt', 'w')
f.write(repr(hist))
f.close()
torch.save(model_ft.state_dict(), 'densenet_model_t8_gc_small.mdl')
