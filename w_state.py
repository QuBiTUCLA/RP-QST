from rp_qst_src import QGAN_method, w_state_3q
import os

#this would have to be changed across computers. There's another that needs to be changed
#at the bottom of QGAN_method() in __init__.py

#I don't know how to make it work universally
snapbase='C:/Users/frank/PycharmProjects/RP-QST/logs/'

kk = [5]
epoch = [500]
batch = [100]

data = w_state_3q()

for i in range(len(kk)):
    for j in range(len(epoch)):
        for k in range(len(batch)):
            snap = str(kk[i])+'_3_'+str(epoch[j])+'_'+str(batch[k])+'_7'
            address = snapbase+snap
            os.mkdir(address)
            QGAN_method(kk[i],3,epoch[j],batch[k],7,address,data)