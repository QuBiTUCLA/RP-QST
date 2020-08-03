from rp_qst_src import QGAN_method, w_state_3q
import os
import numpy as np
import matplotlib.pyplot as plt

snapbase = 'logs/'


kk = [2,3,4,5,6,7]
epoch = [200]
batch = [100]
bounds = 7
num_qubit = 3
models = ['normal','log_normal']

data = w_state_3q()

# last parameter for Q-GAN method is a string to indicate which uncertainty model you want to test
# 'normal' = Normal Distribution
# 'log_normal' = Log-Normal Distribution
# 'uniform' = Uniform Distribution

for i in range(len(kk)):
    for j in range(len(epoch)):
        for k in range(len(batch)):
            for mo in range(len(models)):
                snap = str(kk[i]) + '_' + str(num_qubit) + '_' + str(epoch[j]) + '_' + str(batch[k]) + '_' + str(bounds)\
                       + '_' + str(models[mo])
                address = snapbase+snap
                if os.path.exists(address):
                    continue
                else:
                    os.mkdir(address)
                qgan = QGAN_method(kk[i], num_qubit, epoch[j], batch[k], bounds, address, data, models[mo])

                # Plot progress w.r.t the generator's and the discriminator's loss function
                t_steps = np.arange(epoch[j])
                plt.figure(figsize=(6, 5))
                plt.title("Progress in the loss function - "+ snap)
                plt.plot(t_steps, qgan.g_loss, label="Generator loss function", color='mediumvioletred', linewidth=2)
                plt.plot(t_steps, qgan.d_loss, label="Discriminator loss function", color='rebeccapurple', linewidth=2)
                plt.grid()
                plt.legend(loc='best')
                plt.xlabel('time steps')
                plt.ylabel('loss')

                name = str(kk[i]) + '_' + str(num_qubit) + '_' + str(epoch[j]) + '_' + str(batch[k]) + '_' + str(bounds)\
                       + '_' + str(models[mo]) + '_loss.png'
                address = 'logs/Generator_And_Discriminator_Loss_Plots/' + name
                plt.savefig(address)


                # Plot progress w.r.t relative entropy
                plt.figure(figsize=(6, 5))
                plt.title("Relative Entropy - " + snap)
                plt.plot(np.linspace(0, epoch[j], len(qgan.rel_entr)), qgan.rel_entr, color='mediumblue', lw=4, ls=':')
                plt.grid()
                plt.xlabel('time steps')
                plt.ylabel('relative entropy')

                name = str(kk[i]) + '_' + str(num_qubit) + '_' + str(epoch[j]) + '_' + str(batch[k]) + '_' + str(bounds)\
                       + '_' + str(models[mo]) + '_entr.png'
                address = 'logs/Entropy_Plots/' + name
                plt.savefig(address)


                # Plot the PDF of the resulting distribution against the target distribution, i.e. log-normal
                log_normal = data
                # log_normal = np.round(log_normal)
                # log_normal = log_normal[log_normal <= bounds[1]]
                temp = []
                for l in range(int(bounds + 1)):
                    temp += [np.sum(log_normal == l)]
                log_normal = np.array(temp / sum(temp))

                plt.figure(figsize=(6, 5))
                plt.title("W-State QGAN - "+ snap)
                samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
                samples_g = np.array(samples_g)
                samples_g = samples_g.flatten()
                num_bins = len(prob_g)
                plt.bar(samples_g, prob_g, color='royalblue', width=0.8, label='Simulation')
                plt.plot(log_normal, '-o', label='W-State Measurements', color='deepskyblue', linewidth=4, markersize=12)
                plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
                plt.grid()
                plt.xlabel('states')
                plt.ylabel('p(x)')
                plt.legend(loc='best')

                name = str(kk[i]) + '_' + str(num_qubit) + '_' + str(epoch[j]) + '_' + str(batch[k]) + '_' + str(bounds)\
                       + '_' + str(models[mo]) + '_plot.png'
                address = 'logs/Probability_Plots/' + name
                plt.savefig(address)