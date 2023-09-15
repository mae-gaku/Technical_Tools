import torch
import torchvision



data_dir_path = ''

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((240, 320)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root=data_dir_path + 'data',transform=data_transforms)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=1)

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

# cross-entropy loss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

net.train()

n_epochs = 50
for epoch in range(n_epochs):
    running_loss = 0.0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
    print("Epoch: {}; Loss: {}".format(epoch, running_loss))

print("Finished Training")

# evaltion
net.eval()

test_dataset = torchvision.datasets.ImageFolder(root=data_dir_path + 'TEST',
                                                transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                              num_workers=1)
n_correct = 0
n_total = 0

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        n_total += labels.size(0)
        n_correct += (predicted == labels).sum().item()

print('#images: {}; acc: {}'.format(n_total, 100 * n_correct / n_total))