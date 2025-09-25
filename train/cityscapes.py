import torch
from torch.utils.data import DataLoader

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

from models.ddrnets23slim import DDRNetS23slim, load_weights
from utils.datasets.cityscapes import Cityscapes
from utils.losses.ohem_loss import OhemCrossEntropyLoss


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

# def generate_model_filename(model_name, mean_iou):
#     current_datetime = datetime.datetime.now()

#     formatted_accuracy = f'{mIoU:.1f}'.replace('.', '_')

#     datetime_str = current_datetime.strftime('%y-%m-%d_%H-%M-%S')

#     filename = f'{model_name}_miou_{formatted_accuracy}_{datetime_str}.pth'

#     return filename

# #}


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    configs = read_config_file('../configs/cityscapes.yaml')['train']

    # #{ prepare Cityspaces dataset

    home_path = Path.home()
    root_path = home_path / '../app/data/cityscapes'

    train_dataset = Cityscapes(root=root_path, mode='train')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=configs['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # #}

    ddrnets23slim = DDRNetS23slim(num_classes=19)

    criterion = OhemCrossEntropyLoss()

    optimizer = None
    if configs['optimizer']['name'] == 'SGD':
        optimizer = optim.SGD(
            ddrnets23slim.parameters(),
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

    ddrnets23slim = ddrnets23slim.to(device)

    num_epochs = configs['num_epochs']
    for epoch in range(num_epochs):
        # Set model to training mode
        ddrnets23slim.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = ddrnets23slim(inputs)
                normal_loss = criterion(outputs[0], labels)
                aux_loss = criterion(outputs[1], labels)
                loss = normal_loss + 0.4 * aux_loss
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print('Epoch loss: {:.4f}'.format(epoch_loss))

    print('Training finished!')

    model_filename = 'ddrnets23slim_dummy.pth'
    torch.save(ddrnets23slim.state_dict(), model_filename)
    print(f'Weights saved to {model_filename}!')
