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
parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu_id', type=str, default='0')
opt = parser.parse_args()

torch.manual_seed(opt.seed)

config = get_config(opt.config)
device = torch.device('cuda:{}'.format(opt.gpu_id) if config['cuda'] else 'cpu')


if config['task'] == 'mnist':
	test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST', 
							train=False, transform=transforms.ToTensor()),
							batch_size=config['batch_size_test'], shuffle=False, num_workers=4, drop_last=True)
	config[f"vae{config['model']}"]['channels'], config[f"vae{config['model']}"]['width'], config[f"vae{config['model']}"]['height'] = 1, 28, 28

elif config['task'] == 'fmnist':
	test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data/FashionMNIST', 
							train=False, transform=transforms.ToTensor()),
							batch_size=config['batch_size_test'], shuffle=False, num_workers=4, drop_last=True)	

	config[f"vae{config['model']}"]['channels'], config[f"vae{config['model']}"]['width'], config[f"vae{config['model']}"]['height'] = 1, 28, 28

elif config['task'] == 'cifar':
	transform_test = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
								]
							)


	test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root = './data/CIFAR10', train = False,
							download = True, transform = transform_test),
							batch_size=config['batch_size_test'], shuffle=False, num_workers=4, drop_last=True)

	config[f"vae{config['model']}"]['channels'], config[f"vae{config['model']}"]['width'], config[f"vae{config['model']}"]['height'] = 3, 32, 32

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

	test_loader = torch.utils.data.DataLoader(MyCelebA(root = './data/CelebA', split = 'train',
							download = False, transform = transform_train),
							batch_size=config['batch_size_train'], shuffle=True, num_workers=4, drop_last=True)


else:
	assert 0, f"Not Implemented {config['task']}"




kl_beta = np.logspace(-2, 1, 50)
results = {'score_det': [], 
			'score_trace': [],
			'von_entr': [], 
			'lambda_max': [],
			'S_norm': [],
			'error': [],
			}

output_results = config['output_results']
output_model = config['output_model']
steps = np.logspace(-1, 1, 40)
epoch = 100
robust = []
prone = []


config['kl_term'] = 1
config['output_results'] = f"/disk/scratch1/asif/workspace/RobustnessGeometryVAE/model/mixup/{config['task']}/{config['model']}kl_{config['kl_term']}"
config['output_model'] = f"/disk/scratch1/asif/workspace/RobustnessGeometryVAE/model/mixup/kl_{config['kl_term']}" 

if not os.path.exists(config['output_results']):
	os.makedirs(config['output_results'])
trainer = VAETrainer(config, device)
ckpt = torch.load(f"{config['output_model']}/model_{config['model']}task_{config['task']}_beta_{config['kl_term']}vae_{epoch}.pt")
trainer.vae.load_state_dict(ckpt['vae'])
trainer.optG.load_state_dict(ckpt['optG'])

for i, (data, label) in enumerate(test_loader, 0):
	data = data.to(device)
	error_eigen, entr, maxeigen = trainer.sample_eigen(data, config['output_results'], i)
	results['von_entr'].append(entr)
	results['lambda_max'].append(maxeigen)
	results['error'].append(error_eigen)
	if i==5:
		break

results['von_entr'] = torch.stack(results['von_entr'], 0).numpy().flatten()
results['lambda_max'] = torch.stack(results['lambda_max'], 0).numpy().flatten()
#results['S_norm'][kl] = torch.stack(results['S_norm'][kl], 0).numpy()
results['error'] = torch.stack(results['error'], 0).numpy()

nm_eigen = 5
fig, ax = plt.subplots(figsize=(w, h), dpi=150)
for k in range(nm_eigen):
	ax.plot(steps, results['error'][:,k], label=fr"$\lambda_{k+1}$")
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
ax.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.12), fancybox=True)
ax.set_xlabel('Step Size')
ax.set_ylabel('MSE')
ax.set_title(fr"$\beta={kl:.4f}$")
plt.savefig(f"{config['output_results']}/mixup_{config['mixup']}error_avg_batch_{kl}_{task}.png")
plt.cla()
plt.close()


results_mixup = {'score_det': [], 
			'score_trace': [],
			'von_entr': [], 
			'lambda_max': [],
			'S_norm': [],
			'error': [],
			}


config['mixup'] = False
config['kl_term'] = 1
config['output_results'] = f"/disk/scratch1/asif/workspace/RobustnessGeometryVAE/model/mixup/{config['task']}/{config['model']}kl_{config['kl_term']}"
config['output_model'] = f"/disk/scratch1/asif/workspace/RobustnessGeometryVAE/model/mixup/kl_{config['kl_term']}" 
trainer = VAETrainer(config, device)
ckpt = torch.load(f"{config['output_model']}/model_{config['model']}task_{config['task']}_beta_{config['kl_term']}vae_{epoch}.pt")
trainer.vae.load_state_dict(ckpt['vae'])
trainer.optG.load_state_dict(ckpt['optG'])

for i, (data, label) in enumerate(test_loader, 0):
	data = data.to(device)
	error_eigen, entr, maxeigen = trainer.sample_eigen(data, config['output_results'], i)
	results_mixup['von_entr'].append(entr)
	results_mixup['lambda_max'].append(maxeigen)
	results_mixup['error'].append(error_eigen)
	if i==5:
		break

results_mixup['von_entr'] = torch.stack(results['von_entr'], 0).numpy().flatten()
results_mixup['lambda_max'] = torch.stack(results['lambda_max'], 0).numpy().flatten()
#results['S_norm'][kl] = torch.stack(results['S_norm'][kl], 0).numpy()
results_mixup['error'] = torch.stack(results['error'], 0).numpy()


nm_eigen = 5
fig, ax = plt.subplots(figsize=(w, h), dpi=150)
for k in range(nm_eigen):
	ax.plot(steps, results['error'][:,k], label=fr"$\lambda_{k+1}$")
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
ax.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.12), fancybox=True)
ax.set_xlabel('Step Size')
ax.set_ylabel('MSE')
ax.set_title(fr"$\beta={kl:.4f}$")
plt.savefig(f"{config['output_results']}/mixup_{config['mixup']}error_avg_batch_{kl}_{task}.png")
plt.cla()
plt.close()

plt.figure(figsize=(w,h))
plt.hist(results_mixup['lambda_max'], 80, alpha=0.75, density=True, label="mixup")
plt.hist(results['lambda_max'], 80, alpha=0.75, label="no-mixup")
plt.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.15), fancybox=True)
plt.xlabel(r'$\rho (G)$ (Spectral Radius)')
plt.ylabel('Frequency')
plt.title(f"{config['task']}")
plt.xlim(0, 10)
plt.savefig(f"{config['output_results']}/histogram_lambda_max_{kl}_{task}.png")
plt.cla()
plt.close()

plt.figure(figsize=(w,h))
plt.hist(results_mixup['lambda_max'], 80, alpha=0.75, density=True, label="mixup")
plt.hist(results['von_entr'], 80, alpha=0.75, density=True, label="no-mixup")
plt.xlabel(r'$\mathbf{S}$ (Von Neumann Entropy)')
plt.ylabel('Frequency')
plt.xlim(0, 10)
plt.title(f"{config['task']}")
plt.savefig(f"{output_results}/histogram_Entropy_{kl}_{task}.png")
plt.cla()
plt.close()

np.savez(f"{output_results}/{config['task']}/{config['model']}mixup_{config['mixup']}.npz", results['von_entr'][kl], results['lambda_max'][kl], results['error'][kl])