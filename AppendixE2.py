import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('PDF')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
font = {
    'family': 'serif',
    'serif': 'Times New Roman',
    'weight': 'normal',
    'size': 20
}
plt.rc('font',**font)

import os
import pickle
import random
from xmlrpc.client import MAXINT
from cv2 import log
from numpy.core.fromnumeric import sort
import matplotlib.pyplot as plt
from numpy.random.mtrand import f
import numpy as np
from multiprocessing.pool import ThreadPool
import pandas

def read_result(dir):
    res = pandas.read_csv(dir)
    kws_leak_percent = res['kws_leak_percent']
    kws_leak_percent = np.array(kws_leak_percent[:11])
    print(kws_leak_percent)

    Decoding_accuracy = res['Decoding_accuracy']
    Decoding_injection_length = res['Decoding_injection_length']
    Decoding_injection_size = res['Decoding_injection_size']

    BVA_accuracy = res['BVA_accuracy']
    BVA_injection_length = res['BVA_injection_length']
    BVA_injection_size = res['BVA_injection_size']

    BVMA_accuracy = res['BVMA_accuracy']
    BVMA_injection_length = res['BVMA_injection_length']
    BVMA_injection_size = res['BVMA_injection_size']

    SR_1_accuracy = res['SR_accuracy']
    SR_1_injection_length = res['SR_injection_length']
    SR_1_injection_size = res['SR_injection_size']

    SR_linear_accuracy = res['SR_accuracy']
    SR_linear_injection_length = res['SR_injection_length']
    SR_linear_injection_size = res['SR_injection_size']

    return Decoding_accuracy, Decoding_injection_length, Decoding_injection_size, BVA_accuracy, BVA_injection_length, BVA_injection_size,BVMA_accuracy, BVMA_injection_length, BVMA_injection_size,SR_1_accuracy, SR_1_injection_length, SR_1_injection_size, SR_linear_accuracy, SR_linear_injection_length, SR_linear_injection_size

E_L_W = ['Enron']#, 'Lucene', 'Wiki']


for ds in E_L_W:
    x_name = ('no defense', 'x=2', 'x=4')
    Acc_nopadding = pd.read_csv(r'C:\Users\admin\yan\BV(M)A\EnronKwsLeak.csv')
    Acc_padding2 = pd.read_csv(r'C:\Users\admin\yan\BV(M)A\EnronPadding2KwsLeak.csv')
    Acc_padding4 = pd.read_csv(r'C:\Users\admin\yan\BV(M)A\EnronPadding4KwsLeak.csv')


    BVA_avr_acc = []
    BVA_min_acc = []
    BVA_max_acc = []   
    #BVA_recover_rate = Acc_nopadding['BVA_accuracy']
    #BVA_recover_rate = np.array(BVA_recover_rate).reshape(9, 11)
    #BVA_recover_rate = BVA_recover_rate[:, len(BVA_recover_rate[0])-1]
    BVA_recover_rate = [0.72425, 0.8208749999999998, 0.8772125000000001, 0.9072999999999999, 0.9308, 0.9399625, 0.9384, 0.948625, 0.9475874999999998]
    BVA_avr_acc.append(np.mean(BVA_recover_rate, axis=0))
    BVA_min_acc.append(BVA_avr_acc[0] - np.min(BVA_recover_rate, axis=0))
    BVA_max_acc.append(np.max(BVA_recover_rate, axis=0) - BVA_avr_acc[0])
    #BVA_recover_rate = Acc_padding2['BVA_accuracy']
    #BVA_recover_rate = np.array(BVA_recover_rate).reshape(9, 11)
    #BVA_recover_rate = BVA_recover_rate[:, len(BVA_recover_rate[0])-1]
    BVA_recover_rate = [0.7269625, 0.8151499999999998, 0.8788875, 0.9149499999999999, 0.9287124999999999, 0.9410249999999998, 0.9436874999999999, 0.9461750000000002, 0.9461499999999999]
    BVA_avr_acc.append(np.mean(BVA_recover_rate, axis=0))
    BVA_min_acc.append(BVA_avr_acc[1] - np.min(BVA_recover_rate, axis=0))
    BVA_max_acc.append(np.max(BVA_recover_rate, axis=0) - BVA_avr_acc[1])
    #BVA_recover_rate = Acc_padding4['BVA_accuracy']
    #BVA_recover_rate = np.array(BVA_recover_rate).reshape(9, 11)
    #BVA_recover_rate = BVA_recover_rate[:, len(BVA_recover_rate[0])-1]
    BVA_recover_rate = [0.723625, 0.8164375000000001, 0.87905, 0.9114375000000001, 0.9330999999999999, 0.9418000000000001, 0.945, 0.9467625, 0.9478625]
    BVA_avr_acc.append(np.mean(BVA_recover_rate, axis=0))
    BVA_min_acc.append(BVA_avr_acc[1] - np.min(BVA_recover_rate, axis=0))
    BVA_max_acc.append(np.max(BVA_recover_rate, axis=0) - BVA_avr_acc[1])
    print(BVA_avr_acc)
    print(BVA_min_acc)
    print(BVA_max_acc)

    BVMA_avr_acc = []
    #BVMA_recover_rate = Acc_nopadding['BVMA_accuracy']
    #BVMA_recover_rate = np.array(BVMA_recover_rate).reshape(9, 11)
    #BVMA_recover_rate = BVMA_recover_rate[:, len(BVMA_recover_rate[0])-1]  
    BVMA_recover_rate = [0.8939874999999999, 0.89435, 0.8902125, 0.8929375, 0.89205, 0.8895250000000001, 0.8943375000000001, 0.891675, 0.8969500000000001]
    BVMA_avr_acc.append(np.mean(BVMA_recover_rate, axis=0))
    #BVMA_recover_rate = Acc_padding2['BVMA_accuracy']
    #BVMA_recover_rate = np.array(BVMA_recover_rate).reshape(9, 11)
    #BVMA_recover_rate = BVMA_recover_rate[:, len(BVMA_recover_rate[0])-1]
    BVMA_recover_rate = [0.8943749999999999, 0.8926750000000002, 0.8970749999999998, 0.8911374999999999, 0.8882875, 0.8923500000000001, 0.8947625, 0.8886375000000001, 0.8910500000000001]
    BVMA_avr_acc.append(np.mean(BVMA_recover_rate, axis=0))

    #BVMA_recover_rate = Acc_padding4['BVMA_accuracy']
    #BVMA_recover_rate = np.array(BVMA_recover_rate).reshape(9, 11)
    #BVMA_recover_rate = BVMA_recover_rate[:, len(BVMA_recover_rate[0])-1]
    BVMA_recover_rate = [0.8939874999999999, 0.89435, 0.8902125, 0.8929375, 0.89205, 0.8895250000000001, 0.8943375000000001, 0.891675, 0.8969500000000001]
    BVMA_avr_acc.append(np.mean(BVMA_recover_rate, axis=0))
    print(BVMA_avr_acc)

    xticks = np.arange(len(x_name))
    fig, ax = plt.subplots()
    ax.bar(xticks, BVA_avr_acc, yerr = [BVA_min_acc, BVA_max_acc], capsize=4, width=0.34, label="BVA", ecolor = 'g',color="lightgreen", edgecolor = "g", linewidth = 0.8)#, hatch = '-'
    ax.bar(xticks+0.34, BVMA_avr_acc, width=0.34, label="BVMA", color="r", edgecolor = "white", linewidth = 0.8, hatch = '/')
    
    ax.set_ylabel('Recovery rate', fontsize = 20)
    plt.xticks(xticks + 0.34/2, x_name)
    ax.legend(fontsize = 20, loc = 'lower left')

    ax.grid()
    plt.tick_params(labelsize=20)


    plt.savefig('PaddingTrend{}.pdf'.format(ds), bbox_inches = 'tight')