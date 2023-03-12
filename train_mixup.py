import argparse
from trainer import VAETrainer
import torch
import pdb
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import yaml
from sklearn.decomposition import PCA
import torchvision

def get_config(config):
	with open(config, 'r') as f:
		return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_mixup.yaml', help='Configuration file')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()

torch.manual_seed(opt.seed)

config = get_config(opt.config)
device = torch.device('cuda:{}'.format(opt.gpu_id) if config['cuda'] else 'cpu')


if config['task'] == 'mnist':
	train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', 
							train=True, transform=transforms.ToTensor()),
							batch_size=config['batch_size_test'], shuffle=False, num_workers=4, drop_last=True)
	config[f"vae{config['task']}"]['channels'], config[f"vae{config['task']}"]['width'], config[f"vae{config['task']}"]['height'] = 1, 28, 28

elif config['task'] == 'fmnist':
	train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/FashionMNIST', 
							train=True, transform=transforms.ToTensor()),
							batch_size=config['batch_size_test'], shuffle=False, num_workers=4, drop_last=True)	

	config[f"vae{config['task']}"]['channels'], config[f"vae{config['task']}"]['width'], config[f"vae{config['task']}"]['height'] = 1, 28, 28

elif config['task'] == 'cifar':
	transform_train = transforms.Compose([transforms.RandomCrop(32, padding = 4),
								transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								]
								)

	transform_test = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								]
							)
	train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root = './data/CIFAR10', train = True,
							download = True, transform = transform_train),
							batch_size=config['batch_size_train'], shuffle=True, num_workers=4, drop_last=True)

	config[f"vae{config['task']}"]['channels'], config[f"vae{config['task']}"]['width'], config[f"vae{config['task']}"]['height'] = 3, 32, 32

elif config['task'] == 'celeba':
	transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
											transforms.CenterCrop(148),
											transforms.Resize(64),
											transforms.ToTensor()
										]
										)

	transform_test = transforms.Compose([transforms.RandomHorizontalFlip(),
											transforms.CenterCrop(148),
											transforms.Resize(64),
											transforms.ToTensor()
										]
										)
	class MyCelebA(datasets.CelebA):
		"""
		A work-around to address issues with pytorch's celebA dataset class.
		Download and Extract
		URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
		"""

		def _check_integrity(self) -> bool:
			return True

	train_loader = torch.utils.data.DataLoader(MyCelebA(root = './data/CelebA', split = 'train',
							download = False, transform = transform_train),
							batch_size=config['batch_size_train'], shuffle=True, num_workers=4, drop_last=True)


else:
	assert 0, f"Not Implemented {config['task']}"




output_results = config['output_results']  
output_model = config['output_model']

config['kl_term'] = 1
config['output_results'] = f"{output_results}/kl_{config['kl_term']}"
config['output_model'] = f"{output_model}/kl_{config['kl_term']}" 
if not os.path.exists(config['output_results']):
	os.makedirs(config['output_results'])
if not os.path.exists(config['output_model']):
	os.makedirs(config['output_model'])
trainer = VAETrainer(config, device)
for epoch in range(config['num_epochs']):
	print(f"Training Epoch {epoch}")
	for i, (data, label) in enumerate(train_loader, 0):
		data = data.to(device)
		trainer.updateVAE(data)
		trainer.summary(i)
		if (epoch+1) % config['save_every'] == 0:
			trainer.save_model(epoch, config['task'], config['model'])
	trainer.scheduler.step()
trainer.writer.close()
