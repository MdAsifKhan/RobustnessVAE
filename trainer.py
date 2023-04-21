import torch.nn as nn
import torch
from model import VAE
import os
import pdb
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.nn.functional as F
from scipy.stats import entropy
import numpy as np
from torch import autograd
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from einops import rearrange
from torch.distributions.beta import Beta

class VAETrainer(nn.Module):
	def __init__(self, config, device):
		super(VAETrainer,self).__init__()
		self.config = config
		self.vae = VAE(self.config[f"vae{config['task']}"], task=config['task']).to(device)
		self.writer = SummaryWriter('{}/summary'.format(self.config['output_results']))
		self.optG = torch.optim.Adam(self.vae.parameters(), lr=self.config[f"vae{config['task']}"]['lr'], 
										betas=(self.config[f"vae{config['task']}"]['beta1'], self.config[f"vae{config['task']}"]['beta2']))
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, T_max=200)
		self.mixupdist = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
		
	def updateVAE(self, x):
		self.optG.zero_grad()

		z, mu, logsigma = self.vae.encode(x)
		x_recon = self.vae.decode(z)
		
		self.recon_loss = self.vae.recon_loss(x, x_recon)
		self.kld = self.vae.vae_loss(mu, logsigma)
		self.loss_vae = self.kld*self.config['kl_term'] + self.recon_loss
		
		if self.config['mixup']:
			x_mix = self.mixup(x)
			z_mix_enc, mu_mix_enc, logsigma_enc = self.vae.encode(x_mix)
			kld_mix = self.vae.vae_loss(mu_mix_enc, logsigma_enc)
			z_mix = self.mixup(z)
			x_mix_dec = self.vae.decode(z_mix)
			loss1 = F.mse_loss(x_mix, x_mix_dec, reduction='sum')
			loss2 = F.mse_loss(z_mix, z_mix_enc, reduction='sum')
			self.loss_vae = self.loss_vae + kld_mix + loss1 + loss2

		self.loss_vae.backward()
		self.optG.step()

	def mixup(self, x):
		idx = torch.randperm(x.size(0), device=x.device, dtype=torch.long)
		x_shuffle = x[idx]
		alpha = self.mixupdist.sample().item()
		x_mix = alpha * x + (1 - alpha) * x_shuffle
		return x_mix

	def score_robustness(self, S):
		self.vae.eval()
		S = torch.abs(S)
		lambda_max = S[:,0]
		S_norm = S/S.max(1)[0].unsqueeze(1)
		von_entr = (S_norm * torch.log(S_norm+1e-10)).sum(1)*(-1)

		return von_entr.detach().cpu(), lambda_max.detach().cpu()

	def sample_eigen(self, x_sample, output_path, bidx):
		self.vae.eval()
		x_recon = self.vae(x_sample)[0]
		eigen_x, eigen_x_recon, eigen_z, error_eigen, pullback, U, S, V = self.attack(x_sample)
		von_entr, lambda_max  = self.score_robustness(S)
		#rows = int(batch**0.5)
		#rows = 4
		steps, eigen, batch, channels, width, height = eigen_x_recon.shape
		rows = int(batch**0.5)
		save_image(x_sample, nrow=rows, fp=f"{output_path}/{self.config['kl_term']}\
											{self.config['task']}_{self.config['model']}_original_batch_{bidx+1}.png")
		save_image(x_recon, nrow=rows, fp=f"{output_path}/{self.config['kl_term']}\
											{self.config['task']}_{self.config['model']}_reconstructed_{bidx+1}.png")
		for step in range(steps):
			for i in range(1):
				save_image(eigen_x_recon[step,i], nrow=rows, fp=f"{output_path}/{self.config['kl_term']}\
												{self.config['task']}_{self.config['model']}_recon_eigendirection_{i+1}_step_{self.steps[step]}_{bidx+1}.png")
				save_image(eigen_x[step,i], nrow=rows, fp=f"{output_path}/{self.config['kl_term']}\
												{self.config['task']}_{self.config['model']}_corrupted_eigendirection_{i+1}_step_{self.steps[step]}_{bidx+1}.png")
		return error_eigen.detach().cpu(), von_entr, lambda_max

	def attack(self, x, nm_eigen=5, samples=32):
		self.vae.eval()
		batch, channels, width, height = x.shape		
		#pullback = self.pull_back(x)
		pullback = self.pull_back_xtox(x)
		reshape_pullback = pullback.view(batch, channels*width*height, -1)
		
		U, S = self.pull_back_eigen(reshape_pullback, some=False, compute_uv=True)
				
		self.steps = np.logspace(-1, 1, 40)
		#self.steps = [self.steps[i] for i in range(len(self.steps)) if i%2==0]
		eigen_x_step, eigen_z_step, eigen_x_recon_step = [], [], []
		x = x.view(-1, channels*width*height)
		error_eigen_steps = []
		for step in self.steps:
			eigen_x, eigen_z, eigen_x_recon, error_eigen = [], [], [], []
			for eigen in range(nm_eigen):
				x_epsilon = x + step * torch.einsum('i, ij -> ij', S[:,eigen], U[:,:,eigen])
				x_epsilon =  x_epsilon.view(-1, channels, width, height)
				z_epsilon = self.vae.encode(x_epsilon)[1] # Use mean estimate
				x_recon = self.vae.decode(z_epsilon)
				eigen_z.append(z_epsilon)
				eigen_x.append(x_epsilon)
				eigen_x_recon.append(x_recon)
				error_eigen.append(F.mse_loss(x, x_recon.view(-1, channels*width*height), reduction='mean').mean())
			error_eigen = torch.stack(error_eigen)
			error_eigen_steps.append(error_eigen)
			eigen_z = torch.stack(eigen_z)
			eigen_x = torch.stack(eigen_x)

			eigen_x_recon = torch.stack(eigen_x_recon)
			eigen_x_step.append(eigen_x)

			eigen_x_recon_step.append(eigen_x_recon)
			eigen_z_step.append(eigen_z)

		# steps x eigen x batch x channels x width x height
		eigen_x_recon_step = torch.stack(eigen_x_recon_step)
		eigen_x_step = torch.stack(eigen_x_step)
		eigen_z_step = torch.stack(eigen_z_step)
		error_eigen_steps = torch.stack(error_eigen_steps)
		return eigen_x_step, eigen_x_recon_step, eigen_z_step, error_eigen_steps, pullback, U, S, V


	def error_z(self, x, samples=32):
		self.vae.eval()
		batch, channels, width, height = x.shape		
		pullback = self.pull_back(x)
		U, S, V = torch.svd(pullback.view(batch, channels*width*height, -1).cpu(), some=False, compute_uv=True)
		z = self.vae.encode(x)
		x = x.view(-1, channels*width*height)
		error_eigen_steps = []
		for step in np.logspace(-1, 1, 100):
			error = step * torch.einsum('i, ij -> ij', S[:,0], U[:,:,0])
			x_epsilon = x + error
			x_epsilon =  x_epsilon.view(-1, channels, width, height)
			z_epsilon = self.vae.encode(x_epsilon)[1] # Use mean estimate
				
			error_z = (z - z_epsilon)**2.sum(1).sqrt()			
			error_eigen_steps = torch.stack([error.norm(dim=1), error_z])

		# batch x steps x 2
		error_eigen_steps = torch.stack(error_eigen_steps, 1)
		return error_eigen_steps


	def visualise_eigenmax(self, dataloader, output_path, device):
		self.vae.eval()

		data_all, lambda_max, labels_all = [], [], []
		for x_sample, y_sample in dataloader:
			x_sample, y_sample = x_sample.to(device), y_sample.to(device)

			pullback = self.pull_back(x_sample)
			s1 = torch.symeig(pullback, eigenvectors=False)
			lambda_max.append(s1[:,0])
			data_all.append(x_sample)
			labels_all.append(y_sample)

		data_all = torch.stack(data_all)
		lambda_max = torch.stack(lambda_max)
		labels_all = torch.stack(labels_all)
		data_all = data_all.view(-1, 784).numpy()
		lambda_max = lambda_max.view(-1, 1)
		from sklearn.decomposition import PCA
		from sklearn.preprocessing import StandardScaler
		pca = PCA(n_components=2)
		data_all = StandardScaler().fit_transform(data_all)
		data_pca = pca.fit_transform(data_all)

		fig, axs = plt.subplots(2, 1)
		axs[0,0].scatter(data_pca[:,0], data_pca[:,1], c=labels_all.flatten().numpy())
		axs[0,0].set_xlabel('PC 1')
		axs[0,0].set_ylabel('PC 2')
		axs[0,0].set_xlabel('PCA Data')
		axs[1,0].scatter(data_pca[:,0], data_pca[:,1], c=lambda_max.flatten().numpy())
		axs[1,0].set_xlabel('PC 1')
		axs[1,0].set_xlabel('PC 1')
		axs[1,0].set_title('lambda max')
		plt.savefig(f"{output_path}/pca_lamdamax.png")


	def pull_back_eigen(self, x, option='ltoi', stochasticG=False):
		b, c, w, h = x.shape
		x = x.requires_grad_(True)
		_, mu, logsig = self.vae.encode(x)

		if option == 'ltoi':
			#pdb.set_trace()
			#mu, logsig = self.vae.mu(h), self.vae.sigma(h)
			if stochasticG:
				sig = torch.exp(0.5*logsig)
				Jxmu, Jxsigma = [], []
				for i in range(mu.shape[1]):
					Jxmu.append(autograd.grad(mu[:,i], x, mu[:,i].data.new(mu[:,i].shape).fill_(1), create_graph=True)[0])
					Jxsigma.append(autograd.grad(sig[:,i], x, sig[:,i].data.new(sig[:,i].shape).fill_(1), create_graph=True)[0])

				Jxmu = torch.stack(Jxmu, -1).detach()
				Jxsigma = torch.stack(Jxsigma, -1).detach()
				Gxz = torch.einsum('bdn,bdm->bnm', Jxmu, Jxmu) + torch.einsum('bdn,bdm->bnm', Jxsigma, Jxsigma)
				S, U = np.linalg.eigh(Gxz.cpu().numpy())
				idx = S.argsort()[::-1]
				U, S = U[:,idx], S[:,idx]
			else:
				Jxmu = []
				for i in range(mu.shape[1]):
					Jxmu.append(autograd.grad(mu[:,i], x, mu[:,i].data.new(mu[:,i].shape).fill_(1), create_graph=True)[0])

				Jxmu = torch.stack(Jxmu, -1).detach()
				U, S, V = torch.svd(Jxmu.view(batch, channels*width*height, -1).cpu(), some=False, compute_uv=True)
				return, U, S
		elif option == 'otoi':
			x_recon = self.vae.decode(mu)
			x_recon = rearrange(x_recon, 'b c w h -> b (c w h)')
			if stochasticG:
				sig = torch.exp(0.5*logsig)
				zJxmu, zJxsigma = [], []
				for i in range(mu.shape[1]):
					zJxmu.append(autograd.grad(x_recon[:,i], mu, x_recon[:,i].data.new(x_recon[:,i].shape).fill_(1), create_graph=True)[0])
					zJxsigma.append(autograd.grad(x_recon[:,i], sig, x_recon[:,i].data.new(x_recon[:,i].shape).fill_(1), create_graph=True)[0])

				zJxmu = torch.stack(zJxmu, -1)
				zJxsigma = torch.stack(zJxsigma, -1)
				zGxz = torch.einsum('bdn,bdm->bnm', zJxmu, zJxmu) + torch.einsum('bdn,bdm->bnm', zJxsigma, zJxsigma)
				zGxz = zGxz.detach()


				Jxmu, Jxsigma = [], []
				for i in range(mu.shape[1]):
					Jxmu.append(autograd.grad(mu[:,i], x, mu[:,i].data.new(mu[:,i].shape).fill_(1), create_graph=True)[0])
					Jxsigma.append(autograd.grad(sig[:,i], x, sig[:,i].data.new(sig[:,i].shape).fill_(1), create_graph=True)[0])

				Jxmu = torch.stack(Jxmu, -1).detach()
				Jxsigma = torch.stack(Jxsigma, -1).detach()

				xGxz = torch.einsum('bdn, bnm, bdm->bnm', zJxmu, zGxz, zJxmu) + torch.einsum('bdn, bnm, bdm->bnm', zJxsigma, zGxz, zJxsigma)
				xGxz = zGxz.detach()

				S, U = np.linalg.eigh(xGxz.detach().cpu().numpy())
				idx = S.argsort()[::-1]
				U, S = U[:,idx], S[:,idx]
			else:
				x_recon = self.vae.decode(mu)
				#sig = torch.exp(0.5*logsig)
				x_recon = rearrange(x_recon, 'b c w h -> b (c w h)')
				#Jxmu, Jxsigma = [], []
				Jxrecon = []
				for i in range(x_recon.shape[1]):
					Jxrecon.append(autograd.grad(x_recon[:,i], x, x_recon[:,i].data.new(x_recon[:,i].shape).fill_(1), create_graph=True)[0])
					#Jxsigma.append(autograd.grad(sig[:,i], x, sig[:,i].data.new(sig[:,i].shape).fill_(1), create_graph=True)[0])

				Jxrecon = torch.stack(Jxrecon, -1).detach()
				
				U, S, V = torch.svd(Jxrecon.view(batch, channels*width*height, -1).cpu(), some=False, compute_uv=True)
				return, U, S
			else:
				assert 0,f"Invalid Option: {option}"
				return None



	def summary(self, i):
		self.writer.add_scalar(f"Iter_Loss/VariationalLoss{self.config['kl_term']}", self.kld, i)
		self.writer.add_scalar(f"Iter_Loss/ReconstructionLoss{self.config['kl_term']}", self.recon_loss, i)
		self.writer.add_scalar(f"Iter_Loss/VAELoss{self.config['kl_term']}", self.loss_vae, i)

	def eval_summary(self, i):
		self.writer.add_scalar(f"Iter_Loss/VariationalLoss{self.config['kl_term']}", self.val_kld, i)
		self.writer.add_scalar(f"Iter_Loss/ReconstructionLoss{self.config['kl_term']}", self.val_recon_loss, i)
		self.writer.add_scalar(f"Iter_Loss/VAELoss{self.config['kl_term']}", self.val_loss_vae, i)


	def reconstruct(self, x):
		self.vae.eval()
		x_recon = self.vae(x)[0]
		self.vae.train()
		return x_recon

	def save_model(self, epoch, task='mnist', net='MLP'):
		output = os.path.join(self.config['output_model'], f"model_{net}task_{task}_beta_{self.config['kl_term']}vae_{epoch+1}.pt")
		torch.save({'vae':self.vae.state_dict(), 'optG':self.optG.state_dict()}, output)

	def resume(self, checkpoint):
		pass

	def update_lr(self):
		self.schedulerG.step()