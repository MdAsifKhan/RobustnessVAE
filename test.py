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
results = {'score_det': {kl:[] for kl in kl_beta}, 
			'score_trace': {kl:[] for kl in kl_beta},
			'von_entr': {kl:[] for kl in kl_beta}, 
			'lambda_max': {kl:[] for kl in kl_beta},
			'S_norm': {kl:[] for kl in kl_beta},
			'error': {kl:[] for kl in kl_beta},
			}
output_results = config['output_results']
output_model = config['output_model']
steps = np.logspace(-1, 1, 40)
epoch = 100
robust = []
prone = []
for kl in kl_beta:
	config['kl_term'] = kl
	config['output_results'] = f"{output_results}/{config['task']}/{config['model']}kl_{config['kl_term']}"
	config['output_model'] = f"{output_model}/kl_{config['kl_term']}" 
	if not os.path.exists(config['output_results']):
		os.makedirs(config['output_results'])
	trainer = VAETrainer(config, device)
	ckpt = torch.load(f"{config['output_model']}/model_{config['model']}task_{config['task']}_beta_{config['kl_term']}vae_{epoch}.pt")
	trainer.vae.load_state_dict(ckpt['vae'])
	trainer.optG.load_state_dict(ckpt['optG'])

	for i, (data, label) in enumerate(test_loader, 0):
		data = data.to(device)
		error_eigen, entr, maxeigen = trainer.sample_eigen(data, config['output_results'], i)
		#det, trace, entr, maxeigen, scaledeigen = trainer.score_robustness(pullback, U, S, V, config['output_results'])
		#results['score_det'][kl].append(det)
		#results['score_trace'][kl].append(trace)
		results['von_entr'][kl].append(entr)
		results['lambda_max'][kl].append(maxeigen)
		#results['S_norm'][kl].append(scaledeigen)
		#fig, ax = plt.subplots()
		#for j in range(error_eigen.size(1)):
		#	ax.plot(steps, error_eigen[:,j].cpu().numpy(), label=fr"$\lambda_{j+1}$")
		#box = ax.get_position()
		#ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
		#ax.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.10), fancybox=True)
		#ax.set_xlabel('Step Size')
		#ax.set_ylabel('MSE')
		#ax.set_xscale('log')
		#ax.set_title(fr"$\beta={kl:.4f}$") 
		#plt.savefig(f"{config['output_results']}/error_batch_{i}.png")
		#plt.cla()
		#plt.close()
		results['error'][kl].append(error_eigen)
		if i==5:
			break
	#results['score_det'][kl] = torch.stack(results['score_det'][kl], 0).numpy().flatten()
	#results['score_trace'][kl] = torch.stack(results['score_trace'][kl], 0).numpy().flatten()
	results['von_entr'][kl] = torch.stack(results['von_entr'][kl], 0).numpy().flatten()
	results['lambda_max'][kl] = torch.stack(results['lambda_max'][kl], 0).numpy().flatten()
	#results['S_norm'][kl] = torch.stack(results['S_norm'][kl], 0).numpy()
	results['error'][kl] = torch.stack(results['error'][kl], 0).numpy()
	#pdb.set_trace()
	#fig, ax = plt.subplots()
	#for k in range(error_eigen.size(1)):
	#	ax.plot(steps, results['error'][kl].mean(0).squeeze()[:,k], label=fr"$\lambda_{k+1}$")
	#box = ax.get_position()
	#ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
	#ax.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.10), fancybox=True)
	#ax.set_xlabel('Step Size')
	#ax.set_ylabel('MSE')
	#ax.set_xscale('log')
	#ax.set_title(fr"$\beta={kl:.4f}$")
	#plt.savefig(f"{config['output_results']}/error_avg_batch_{kl}.png")
	#plt.cla()
	#plt.close()
	#plt.hist(results['score_det'], 50, alpha=0.75)
	#plt.xlabel('Logdet')
	#plt.ylabel('Frequency')
	#plt.savefig(f"{config['output_results']}/Logdet_{kl}.png")
	#plt.cla()
	#plt.clf()
	#plt.hist(results['score_trace'], 50, alpha=0.75)
	#plt.xlabel('Trace')
	#plt.ylabel('Frequency')
	#plt.cla()
	#plt.clf()
	#plt.hist(results['von_entr'], 50, alpha=0.75)
	#plt.xlabel('Von Neumann Entropy')
	#plt.ylabel('Frequency')
	#plt.cla()
	#plt.clf()
	#plt.hist(results['lambda_max'][kl], 80, alpha=0.75)
	#plt.xlabel(r'$\rho(G)$ (Spectral Radius)')
	#plt.ylabel('Frequency')
	#plt.title(fr"$\beta={kl:.4f}$")
	#plt.savefig(f"{config['output_results']}/histogram_lambda_max_{kl}.png")
	#plt.cla()
	#plt.close()
	#plt.hist(results['von_entr'][kl], 80, alpha=0.75)
	#plt.xlabel(r'$\mathbf{S}$ (Von Neumann Entropy)')
	#plt.ylabel('Frequency')
	#plt.title(fr"$\beta={kl:.4f}$")
	#plt.savefig(f"{config['output_results']}/histogram_Entropy_{kl}.png")
	#plt.cla()
	#plt.close()
#robust_entr = []
#robust_lambda = []
#for kl in kl_beta:
##	for eig in eigen:
#	mask = results['error'][kl]>0.1
#	robust_entr = results['von_entr'][kl][mask]
#	robust_lambda = results['lambda_max'][kl][mask]
#	prone_entr = results['von_entr'][kl][not mask]
#	prone_lambda = results['lambda_max'][not mask]
#	plt.hist(robust_entr, 50, alpha=0.75, label='Robust')
#	plt.hist(prone_entr, 50, alpha=0.75, label='')

for kl in kl_beta:
	np.savez(f"{output_results}/{config['task']}/{config['model']}score_{kl}.npz", results['von_entr'][kl], results['lambda_max'][kl], results['error'][kl])

#std_lambda_max = [lbda.std() for _, lbda in results['lambda_max']]

fig, ax = plt.subplots()
ax.fill_between(kl_beta, avg_entr-std_entr, avg_entr+std_entr, alpha=0.2)
ax.errorbar(kl_beta, avg_entr, yerr=std_entr, uplims=True, lolims=True)
#plt.scatter(kl_beta, avg_entr)
ax.set_xlabel(r'Increasing $\beta$')
ax.set_ylabel(r'S\mathbf{S}$ (Von Neumann Entropy)')
plt.savefig(f"{config['output_results']}/entropy_vs_beta.png")
plt.cla()
plt.close()
#plt.scatter(kl_beta, avg_lambda_max)
fig, ax = plt.subplots()
ax.fill_between(kl_beta, avg_lambda_max-std_lambda_max, avg_lambda_max+std_lambda_max, alpha=0.2)
ax.errorbar(kl_beta, avg_lambda_max, yerr=std_lambda_max, uplims=True, lolims=True)  
plt.xlabel(r'Increasing $\beta$')
plt.ylabel(r'$\rho(G)$ (Spectral Radius)$')
plt.savefig(f"{config['output_results']}/lambda_max_vs_beta.png")
plt.cla()
plt.close()
#for kl in kl_beta:
#	np.savez(f"{output_results}/{config['task']}/{config['model']}score_{kl}.npz", results['von_entr'][kl], results['lambda_max'][kl], results['error'][kl])
