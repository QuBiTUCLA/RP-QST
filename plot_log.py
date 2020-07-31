import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Fill in these five parameters to plot the desired log file
depth = [2]
num_qubits = 3
epochs = [100]
batches = [100]
bound = 7


for i in range(len(depth)):
    for j in range(len(epochs)):
        for k in range(len(batches)):
            name = str(depth[i]) + '_' + str(num_qubits) + '_' + str(epochs[j]) + '_' + str(batches[k]) + '_' + str(bound)
            desired_folder = 'logs/' + name

            # check to see if the file exists
            if os.path.exists(desired_folder):
                file = desired_folder + '/output.csv'
                f = open(file)
                csv_f = csv.reader(f)
                next(csv_f)
                next(csv_f)
                epoch = np.zeros(shape=epochs[j])
                loss_d = np.zeros(shape=epochs[j])
                loss_g = np.zeros(shape=epochs[j])
                entropy = np.zeros(shape=epochs[j])


                # extract the data from the file
                m = 0
                for row in csv_f:
                    if row:
                        epoch[m] = row[0]
                        loss_d[m] = row[1]
                        loss_g[m] = row[2]
                        entropy[m] = row[4]
                        m += 1


                # Plot progress w.r.t the generator's and the discriminator's loss function
                t_steps = np.arange(epochs[j])
                plt.figure(figsize=(6, 5))
                plt.title("Progress in the loss function")
                plt.plot(t_steps, loss_g, label="Generator loss function", color='mediumvioletred', linewidth=2)
                plt.plot(t_steps, loss_d, label="Discriminator loss function", color='rebeccapurple', linewidth=2)
                plt.grid()
                plt.legend(loc='best')
                plt.xlabel('time steps')
                plt.ylabel('loss')
                plt.show()
                #plt.savefig('logs/Generator_And_Discriminator_Loss_Plots/'+name+'.png')

                # Plot progress w.r.t relative entropy
                plt.figure(figsize=(6, 5))
                plt.title("Relative Entropy ")
                plt.plot(epoch, entropy, color='mediumblue', lw=4, ls=':')
                plt.grid()
                plt.xlabel('time steps')
                plt.ylabel('relative entropy')
                plt.show()
                #plt.savefig('logs/Entropy_Plots/'+name+'.png')

            else:
                print('Logs with desired parameters do not exist.')
