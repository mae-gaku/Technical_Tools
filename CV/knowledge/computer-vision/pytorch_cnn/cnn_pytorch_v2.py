import torch
import torchvision
import time
import copy

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((240, 320)),
    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.ToTensor()
    
    ])


data_dir_path = ''
image_datasets = torchvision.datasets.ImageFolder(root=data_dir_path, transform=transform)

train_size = int(0.8 * len(image_datasets))
validation_size = len(image_datasets) - train_size

dataset_sizes = {
    "train":train_size,
    "val":validation_size
}

data_train, data_validation = torch.utils.data.random_split(image_datasets, [train_size, validation_size])

train_loader = torch.utils.data.DataLoader(data_train,batch_size=16,shuffle=True)
validation_loader = torch.utils.data.DataLoader(data_validation, batch_size=16, shuffle=False)


dataloaders = {
    "train":train_loader, 
    "val":validation_loader
}

class Net(torch.nn.Module):

    # network
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 57 * 77, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def forward(self, x):

        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 57 * 77)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

net = Net()

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    
    # save the best model during learning
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # training mode
            else:
                model.eval()   # evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net()
net = net.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(net, dataloaders, criterion, optimizer, scheduler, num_epochs=50)