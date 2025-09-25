import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.nn import CrossEntropyLoss

import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm

import datetime
import os
import sys
import yaml

from pathlib import Path

# #{ include this project packages

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

# #}

from models.ddrnetc23slim import DDRNetC23slim, init_weights

# #{ read_config_file()

def read_config_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            return yaml_content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

# #}


# #{ calculate_topk_accuracy()

def calculate_topk_accuracy(outputs, labels, topk=(1,)):
    max_k = max(topk)
    batch_size = labels.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()

    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

# #}


# #{ compute_top_accuracy()

def compute_top_accuracy(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    outputs_tensor = torch.cat(all_outputs)
    labels_tensor = torch.cat(all_labels)

    top1_acc, top5_acc = calculate_topk_accuracy(outputs_tensor, labels_tensor, topk=(1, 5))

    return top1_acc, top5_acc

# #}


# #{ generate_model_filename()

def generate_model_filename(model_name, top1_acc):
    current_datetime = datetime.datetime.now()

    formatted_accuracy = f'{top1_acc.item():.1f}'.replace('.', '_')

    datetime_str = current_datetime.strftime('%y-%m-%d_%H-%M-%S')

    filename = f'{model_name}_top1_acc_{formatted_accuracy}_{datetime_str}.pth'

    return filename

# #}


if __name__ == '__main__':

    configs = read_config_file('../configs/imagenet.yaml')['train']

    # #{ prepare Imagenet dataset

    home_path = Path().home()
    data_dir = home_path / '../app/data/imagenet-val'
    print('Data directory:', data_dir)

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(
        root=data_dir,
        transform=data_transforms
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
        num_workers=4
    )

    num_classes = len(train_dataset.classes)
    print(f'Number of classes in ImageNet: {num_classes}')

    # #}

    ddrnetc23slim = DDRNetC23slim(num_classes=1000)
    init_weights(ddrnetc23slim)

    criterion = CrossEntropyLoss()

    optimizer = None
    if configs['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(
            ddrnetc23slim.parameters(),
            lr=configs['optimizer']['learning_rate'],
            momentum=configs['optimizer']['momentum'],
            weight_decay=configs['optimizer']['weight_decay'],
            nesterov=True
        )
    else:
        print('Error: no valid optimizer')
        exit(1)

    # learning rate reduced by 10 times at epochs 30, 60, and 90
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 60, 90],
        gamma=0.1
    )

    # training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ddrnetc23slim = ddrnetc23slim.to(device)

    num_epochs = configs['num_epochs']
    for epoch in range(num_epochs):
        # Set model to training mode
        ddrnetc23slim.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = ddrnetc23slim(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print('Epoch loss: {:.4f}'.format(epoch_loss))

    print('Training finished!')

    top1_acc, top5_acc = compute_top_accuracy(ddrnetc23slim, train_loader, device)
    print(f'Validation Top-1 Accuracy: {top1_acc.item():.2f}%')
    print(f'Validation Top-5 Accuracy: {top5_acc.item():.2f}%')

    model_filename = generate_model_filename('ddrnetc23slim', top1_acc)
    torch.save(ddrnetc23slim.state_dict(), model_filename)
    print(f'Weights saved to {model_filename}!')
