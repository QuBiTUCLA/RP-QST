import os
import numpy as np
import csv
import matplotlib.pyplot as plt

#Fill in these five parameters to plot the desired log file
depth = 1
num_qubits = 3
epochs = 100
batches = 100
bound = 7

desired_folder = 'logs/' + str(depth) + '_' + str(num_qubits) + '_' + str(epochs) + '_' + str(batches) + '_' + str(bound)

#check to see if the file exists
if os.path.exists(desired_folder):
    file = desired_folder + '/output.csv'
    f = open(file)
    csv_f = csv.reader(f)
    next(csv_f)
    next(csv_f)
    epoch = []
    loss_discriminator = []
    loss_generator = []
    entropy = []

    #extract the data from the file
    for row in csv_f:
        if row:
            epoch.append(row[0])
            loss_discriminator.append(row[1])
            loss_generator.append(row[2])
            entropy.append(row[4])
    print(entropy)

    # Plot progress w.r.t the generator's and the discriminator's loss function
    plt.figure(figsize=(6, 5))
    plt.title("Progress in the loss function")
    plt.plot(epoch, loss_generator, label="Generator loss function", color='mediumvioletred', linewidth=2)
    plt.plot(epoch, loss_discriminator, label="Discriminator loss function", color='rebeccapurple', linewidth=2)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time steps')
    plt.ylabel('loss')
    plt.show()

    # Plot progress w.r.t relative entropy
    plt.figure(figsize=(6, 5))
    plt.title("Relative Entropy ")
    plt.scatter(epoch, entropy, color='mediumblue', lw=4, ls=':')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('relative entropy')
    plt.show()

else:
    print('Logs with desired parameters do not exist.')





