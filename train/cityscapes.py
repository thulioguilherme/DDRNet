import torch
from torch.utils.data import DataLoader

from torchvision.datasets import Cityscapes
from torchvision import transforms

from torch.nn import CrossEntropyLoss

import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm

import datetime
import os
import sys
import yaml

# #{ include this project packages

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

# #}

from models.DDRNetS23slim import DDRNetS23slim


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

# #{ generate_model_filename()

# def generate_model_filename(model_name, top1_acc):
#     current_datetime = datetime.datetime.now()

#     formatted_accuracy = f'{mIoU:.1f}'.replace('.', '_')

#     datetime_str = current_datetime.strftime('%y-%m-%d_%H-%M-%S')

#     filename = f'{model_name}_top1_acc_{formatted_accuracy}_{datetime_str}.pth'

#     return filename

# #}

if __name__=='__main__':

    configs = read_config_file('../configs/cityscapes.yaml')['train']

    # #{ prepare Cityspaces dataset

    home_dir = os.path.expanduser('~')
    data_dir = os.path.join(home_dir, '../app/data/cityscapes')
    print('Data directory:', data_dir)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Cityscapes(
        root=root_dir,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
        num_workers=4
    )

    num_classes = len(train_dataset.classes)
    print(f'Number of classes in Cityscapes: {num_classes}')

    # #}

    ddrnets = DDRNetS23slim(num_classes=19)

    criterion = CrossEntropyLoss()

    optimizer = None
    if configs['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(
            ddrnetc.parameters(),
            lr=configs['optimizer']['learning_rate'],
            momentum=configs['optimizer']['momentum'],
            weight_decay=configs['optimizer']['weight_decay'],
            nesterov=True
        )
    else:
        print('Error: no valid optimizer')
        exit(1)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.94
    )

    # Training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ddrnets = ddrnets.to(device)

    num_epochs = configs['num_epochs']
    for epoch in range(num_epochs):
        # Set model to training mode
        ddrnets.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = ddrnetc(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print('Epoch loss: {:.4f}'.format(epoch_loss))

    print('Training finished!')

    model_filename = generate_model_filename('ddrnets', top1_acc)
    torch.save(ddrnets.state_dict(), model_filename)
    print(f'Weights saved to {model_filename}!')
