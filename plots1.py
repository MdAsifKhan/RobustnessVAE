import numpy as np
import matplotlib.pyplot as plt
import pdb

#kl_10.0
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)
kl_beta = np.logspace(-2, 1, 50)
steps = np.logspace(-1, 1, 40)
w, h = 7, 6
nm_eigen = 5
output_results = '/home/asif/workspace/RobustnessGeometryVAE/results/pullxmnist'
task = 'mnist'
#model = 'MLPkl_10.0'
model = 'MLP'
entr_all, lambda_max_all, error_all = [], [], {eigen:[] for eigen in range(nm_eigen)}
fig, ax = plt.subplots(figsize=(w, h), dpi=150)
kl_plots = [kl_beta[0], kl_beta[10], kl_beta[32], kl_beta[40], kl_beta[45]]
for kl in kl_plots:
	data = np.load(f"{output_results}/{model}score_{kl}.npz")
	lambda_max = data['arr_1']
	lambda_max[lambda_max<0] = 0
	ax.hist(lambda_max, 80, alpha=0.75, label=fr"$\beta={kl:.3f}$")
box = ax.get_position()
ax.set_position([box.x0 , box.y0+box.height*0.2, box.width*0.9, box.height*0.8])
ax.legend(loc='upper center', shadow=True, ncol=3, bbox_to_anchor=(0.5, -0.15), fancybox=True)
ax.set_xlabel(r'$\rho (G)$ (Spectral Radius)')
ax.set_ylabel('Frequency')
ax.set_title(f"Spectral Radius ({task.upper()})")
plt.xlim(0, 10)
plt.savefig(f"{output_results}/histogram_lambda_max_{task}.png")

plt.cla()
plt.clf()

fig, ax = plt.subplots(figsize=(w, h), dpi=150)
for kl in kl_plots:
	data = np.load(f"{output_results}/{model}score_{kl}.npz")
	von_entr = data['arr_0']
	ax.hist(von_entr, 80, alpha=0.75, label=fr"$\beta={kl:.3f}$")
box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.2, box.width*0.9, box.height*0.8])
ax.legend(loc='upper center', shadow=True, ncol=3, bbox_to_anchor=(0.5, -0.15), fancybox=True)
ax.set_xlabel(r'$\mathbf{S}$ (Von Neumann Entropy)')
ax.set_ylabel('Frequency')
ax.set_title(f"Von Neumann Entropy ({task.upper()})")
plt.xlim(0, 10)
plt.savefig(f"{output_results}/histogram_von_neumann_{task}.png")


avg_entr, std_entr, avg_lambda_max, std_lambda_max, min_lambda = [], [], [], [], []
for kl in kl_beta:
	data = np.load(f"{output_results}/{model}score_{kl}.npz")
	von_entr = data['arr_0']
	lambda_max = data['arr_1']
	avg_entr.append(von_entr.mean())
	std_entr.append(von_entr.std())
	avg_lambda_max.append(np.median(lambda_max))
	std_lambda_max.append(lambda_max.min())

avg_entr = np.array(avg_entr)
std_entr = np.array(std_entr)
avg_lambda_max = np.array(avg_lambda_max)
std_lambda_max = np.array(std_lambda_max)
fig, ax = plt.subplots(figsize=(w, h), dpi=150)
ax.fill_between(kl_beta, avg_entr-std_entr, avg_entr+std_entr, alpha=0.2)
ax.errorbar(kl_beta, avg_entr, yerr=std_entr, uplims=True, lolims=True)
#plt.scatter(kl_beta, avg_entr)
ax.set_xlabel(r'Increasing $\beta$')
ax.set_ylabel(r'$\mathbf{S}$ (Von Neumann Entropy)')
ax.set_xscale('log')
ax.set_title(f"{task.upper()}")
plt.savefig(f"{output_results}/entropy_vs_beta_{task}.png")
plt.cla()
plt.close()
#plt.scatter(kl_beta, avg_lambda_max)
fig, ax = plt.subplots(figsize=(w, h), dpi=150)
ax.fill_between(kl_beta, avg_lambda_max-std_lambda_max, avg_lambda_max+std_lambda_max, alpha=0.2)
ax.errorbar(kl_beta, avg_lambda_max, yerr=std_lambda_max, uplims=True, lolims=True)  
ax.set_xscale('log')
ax.set_xlabel(r'Increasing $\beta$')
ax.set_ylabel(r'$\rho (G)$ (Spectral Radius)')
ax.set_title(f"{task.upper()}")
plt.savefig(f"{output_results}/lambda_max_vs_beta_{task}.png")
plt.cla()
plt.close()