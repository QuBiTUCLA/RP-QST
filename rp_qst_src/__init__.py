#!/usr/bin/env python
# coding: utf-8

from qiskit import Aer
from qiskit import ClassicalRegister, execute

# packages for QGAN
import numpy as np


import time

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.uncertainty_models import UniformDistribution, UnivariateVariationalDistribution
from qiskit.aqua.components.variational_forms import RY

from qiskit.aqua.algorithms import QGAN
from qiskit.aqua.components.neural_networks import NumPyDiscriminator

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.initial_states import Custom

from qiskit import BasicAer



# In[17]:


# Here, two useful routine
# Define a F_gate
def F_gate(circ, q, i, j, n, k):
    theta = np.arccos(np.sqrt(1 / (n - k + 1)))
    circ.ry(-theta, q[j])
    circ.cz(q[i], q[j])
    circ.ry(theta, q[j])
    circ.barrier(q[i])


# Define the cxrv gate which uses reverse CNOT instead of CNOT
def cxrv(circ, q, i, j):
    circ.h(q[i])
    circ.h(q[j])
    circ.cx(q[j], q[i])
    circ.h(q[i])
    circ.h(q[j])
    circ.barrier(q[i], q[j])


# In[38]:

def w_state_3q():
    "Choice of the backend"
    # using local qasm simulator
    backend = Aer.get_backend('qasm_simulator')

    # using IBMQ qasm simulator
    # backend = IBMQ.get_backend('ibmq_qasm_simulator')
    # using real device
    # backend = least_busy(IBMQ.backends(simulator=False))

    flag_qx2 = True
    if backend.name() == 'ibmqx4':
        flag_qx2 = False

    print("Your choice for the backend is: ", backend, "flag_qx2 is: ", flag_qx2)

    # 3-qubit W state
    n = 3
    q = QuantumRegister(n)
    c = ClassicalRegister(n)

    W_states = QuantumCircuit(q, c)

    W_states.x(q[2])  # start is |100>
    F_gate(W_states, q, 2, 1, 3, 1)  # Applying F12
    F_gate(W_states, q, 1, 0, 3, 2)  # Applying F23

    if flag_qx2:  # option ibmqx2
        W_states.cx(q[1], q[2])  # cNOT 21
        W_states.cx(q[0], q[1])  # cNOT 32

    else:  # option ibmqx4
        cxrv(W_states, q, 1, 2)
        cxrv(W_states, q, 0, 1)

    for i in range(3):
        W_states.measure(q[i], c[i])

    shots = 1000
    time_exp = time.strftime('%d/%m/%Y %H:%M:%S')
    print('start W state 3-qubit on', backend, "N=", shots, time_exp)
    result = execute(W_states, backend=backend, shots=shots)
    time_exp = time.strftime('%d/%m/%Y %H:%M:%S')
    print('end   W state 3-qubit on', backend, "N=", shots, time_exp)

    # In[39]:


    frequencies = result.result().get_counts(W_states)
    freq1 = frequencies['001']
    freq2 = frequencies['010']
    freq3 = frequencies['100']

    line1 = [1]
    line2 = [2]
    line3 = [4]

    real_data = np.zeros(shape=1000)  # real_data is actually an ndarray, not an array



    i = 0
    linespassed = 0
    for i in range(freq1):
        real_data[i] += line1

    linespassed += freq1
    for i in range(freq2):
        real_data[linespassed + i] += line2

    linespassed += freq2
    for i in range(freq3):
        real_data[linespassed + i] += line3


    return real_data


def QGAN_method(kk,num_qubit,epochs,batch,bound,snap, data):
    start = time.time()

    real_data = data

    # In[41]:


    # Number training data samples
    N = 1000

    # Load data samples from log-normal distribution with mean=1 and standard deviation=1
    mu = 1
    sigma = 1

    # real_data = np.random.lognormal(mean = mu, sigma=sigma, size=N)
    # print(real_data)

    # Set the data resolution
    # Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
    bounds = np.array([0,bound])

    # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
    num_qubits = [num_qubit]

    k = kk

    # In[52]:


    # Set number of training epochs
    # Note: The algorithm's runtime can be shortened by reducing the number of training epochs.
    num_epochs = epochs
    # Batch size
    batch_size = batch

    # Initialize qGAN
    qgan = QGAN(real_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=snap)
    qgan.seed = 1
    # Set quantum instance to run the quantum generator
    quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'))

    # Set entangler map
    entangler_map = [[0, 1]]

    # Set an initial state for the generator circuit
    init_dist = UniformDistribution(sum(num_qubits), low=bounds[0], high=bounds[1])
    q = QuantumRegister(sum(num_qubits), name='q')
    qc = QuantumCircuit(q)
    init_dist.build(qc, q)
    init_distribution = Custom(num_qubits=sum(num_qubits), circuit=qc)
    var_form = RY(int(np.sum(num_qubits)), depth=k, initial_state=init_distribution,
                  entangler_map=entangler_map, entanglement_gate='cz')

    # Set generator's initial parameters
    init_params = aqua_globals.random.rand(var_form._num_parameters) * 2 * np.pi
    # Set generator circuit
    g_circuit = UnivariateVariationalDistribution(int(sum(num_qubits)), var_form, init_params,
                                                  low=bounds[0], high=bounds[1])
    # Set quantum generator
    qgan.set_generator(generator_circuit=g_circuit)
    # Set classical discriminator neural network
    discriminator = NumPyDiscriminator(len(num_qubits))
    qgan.set_discriminator(discriminator)

    # In[53]:


    # Run qGAN
    qgan.run(quantum_instance)

    # Runtime
    end = time.time()
    print('qGAN training runtime: ', (end - start) / 60., ' min')

    return qgan
