import numpy as np
import matplotlib.pyplot as plt

#kl_10.0
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)
kl_beta = np.logspace(-2, 1, 50)
steps = np.logspace(-1, 1, 40)
nm_eigen = 5
output_results = '/home/asif/workspace/RobustnessGeometryVAE/results/pullxfmnist'
task = 'fmnist'
#model = 'MLPkl_10.0'
model = 'MLP'
w, h = 7, 6
entr_all, lambda_max_all, error_all = [], [], {eigen:[] for eigen in range(nm_eigen)}
for kl in kl_beta:
	data = np.load(f"{output_results}/{model}score_{kl}.npz")
	von_entr = data['arr_0']
	lambda_max = data['arr_1']
	error = data['arr_2']

	fig, ax = plt.subplots(figsize=(w, h), dpi=150)
	for k in range(nm_eigen):
		ax.plot(steps, error.sum(0)[:,k], label=fr"$\lambda_{k+1}$")
	box = ax.get_position()
	ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
	ax.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.12), fancybox=True)
	ax.set_xlabel('Step Size')
	ax.set_ylabel('MSE')
	ax.set_title(fr"$\beta={kl:.4f}$")
	plt.savefig(f"{output_results}/error_avg_batch_{kl}_{task}.png")
	plt.cla()
	plt.close()

	plt.figure(figsize=(w,h))
	plt.hist(lambda_max, 80, alpha=0.75, density=True)
	plt.xlabel(r'$\rho (G)$ (Spectral Radius)')
	plt.ylabel('Frequency')
	plt.title(fr"$\beta={kl:.4f}$")
	plt.xlim(0, 10)
	plt.savefig(f"{output_results}/histogram_lambda_max_{kl}_{task}.png")
	plt.cla()
	plt.close()
	plt.figure(figsize=(w,h))
	plt.hist(von_entr, 80, alpha=0.75, density=True)
	plt.xlabel(r'$\mathbf{S}$ (Von Neumann Entropy)')
	plt.ylabel('Frequency')
	plt.xlim(0, 10)
	plt.title(fr"$\beta={kl:.4f}$")
	plt.savefig(f"{output_results}/histogram_Entropy_{kl}_{task}.png")
	plt.cla()
	plt.close()

	entr_all.append(von_entr.mean())
	lambda_max_all.append(lambda_max.mean())
	for eigen in range(nm_eigen):
		error_all[eigen].append(error.sum(0)[:, eigen])

error_all = {eigen:np.array(err)  for eigen, err in error_all.items()}
for i, step in enumerate(steps):
	fig, ax = plt.subplots(figsize=(w, h), dpi=150)
	for eigen in range(nm_eigen):	
		ax.plot(kl_beta, error_all[eigen][:,i], label=fr"$\lambda_{eigen+1}$")
	box = ax.get_position()
	ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
	ax.legend(loc='upper center', shadow=True, ncol=5, bbox_to_anchor=(0.5, -0.15), fancybox=True)
	ax.set_xlabel(r'Increasing $\beta$')
	#ax.set_ylim([0, 0.25])
	ax.set_ylabel('MSE')
	#ax.set_xscale('log')
	ax.set_title(fr"step$={step:.4f}$")
	plt.savefig(f"{output_results}/error_avg_step_{step:.4f}_{task}.png")
	plt.cla()
	plt.close()
# plt.figure(figsize=(w,h))
# plt.scatter(kl_beta, entr_all)
# plt.xlabel(r'Increasing $\beta$')
# plt.ylabel(r'$\mathbf{S}$ (Von Neumann Entropy)')
# plt.xscale('log')
# plt.savefig(f"{output_results}/{task}/entropy_vs_beta_{task}.png")
# plt.cla()
# plt.close()
# plt.figure(figsize=(w,h))
# plt.scatter(kl_beta, lambda_max_all)
# plt.xlabel(r'Increasing $\beta$')
# plt.ylabel(r'$\rho (G)$ (Spectral Radius)$')
# plt.xscale('log')
# plt.savefig(f"{output_results}/{task}/lambda_max_vs_beta_{task}.png")
# plt.cla()
# plt.close()