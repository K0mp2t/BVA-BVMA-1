import math
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import utils

class BVMA: 
    
    def __init__(self, chosen_kws, T, m, gamma, offset_of_Decoding):
        self.chosen_kws = chosen_kws

        self.inject_file_face_TC = 0
        self.T = T
        self.m = m
        self.gamma = gamma
        """
        get offset
        """
        self.offset = offset_of_Decoding 

    def Decoding_inject(self):
        self.inject_file_face_TC = 0
        for kw_id in range(len(self.chosen_kws)):
            self.inject_file_face_TC += math.ceil(kw_id*self.offset/self.T)

    def BVA_inject(self):   
        self.inject_file_face_TC = 0
        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        """
        generate injected doc
        """
        size_each_doc = []
        if num_injection_doc >= 1:
            size_each_doc.append(self.gamma)
        if num_injection_doc >= 2:
            for i in range(1, num_injection_doc):
                size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1])

        for s in size_each_doc:
            last_part_size = 0 #Ref_F2
            if s%t!=0:
                last_part_size = math.ceil(self.T/(s%self.T))
            self.inject_file_face_TC += math.ceil(len(self.chosen_kws)*0.5/self.T)*(math.floor(s/self.T)+last_part_size)    

    def BVMA_inject(self):
        
        self.inject_file_face_TC = 0

        kws_each_doc = math.ceil(len(self.chosen_kws)/2)
        if kws_each_doc==0:
            num_injection_doc=0
        else:
            num_injection_doc = math.ceil(np.log2(kws_each_doc + kws_each_doc))
        """
        generate doc.
        """
        size_each_doc = [] 
        if num_injection_doc >= 1:
            size_each_doc.append(1 + (int) (len(self.chosen_kws)/2))
        if num_injection_doc >= 2:
            for i in range(1, num_injection_doc):
                size_each_doc.append(size_each_doc[i-1] + size_each_doc[i-1] - (int) (len(self.chosen_kws)/2))

        for s in size_each_doc:
            last_part_size = 0
            if s%t!=0:
                last_part_size = math.ceil(self.T/(s%self.T))
            self.inject_file_face_TC += math.ceil(len(self.chosen_kws)*0.5/self.T)*(math.floor(s/self.T)+last_part_size) 

    def SingleRound_inject(self):
        self.inject_file_face_TC = 0
        for i in range((int) (len(self.chosen_kws))):
            self.inject_file_face_TC += math.ceil((i+1)/self.T)*self.m
        
    def ZKP_inject(self):
        self.inject_file_face_TC = 0
        number_kws = len(self.chosen_kws)
        if self.T>=number_kws/2:
            self.inject_file_face_TC = math.ceil(math.log2(number_kws))
        else:
            self.inject_file_face_TC = math.ceil(number_kws*0.5/self.T)*(math.ceil(math.log2(2*self.T))+1)-1

if __name__=='__main__':
    
    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ choose dataset """
    d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3.WikiPedia ")
    dataset_name = ''
    print(d_id)
    if d_id=='1':
        dataset_name = 'Enron'
    elif d_id=='2':
        dataset_name = 'Lucene'  
    elif d_id=='3':
        dataset_name = 'Wiki'
    else:
        raise ValueError('No Selected Dataset!!!')

    """ read data """
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    with open(os.path.join(utils.DATASET_PATH,"{}_wl_v_off.pkl".format(dataset_name, dataset_name.lower())), "rb") as f:
        _, _, offset_of_Decoding = pickle.load(f)
        f.close()
    SingleRound_l = []
    Decoding_l = []
    BVA_l = []
    BVMA_l = []
    ZKP_l = []
    T_list = []
    now_T = 2
    mulp = 4
    if d_id==3:
        mulp = 8
    while(now_T<len(chosen_kws)):
        T_list.append(now_T)
        now_T *= mulp
    T_list.append(len(chosen_kws))
    m_of_SingleRound = (int) (len(chosen_kws)/2)
    gamma_of_BVA = (int) (len(chosen_kws)/2)
    for t in T_list:
        attack = BVMA(chosen_kws, t, m_of_SingleRound, gamma_of_BVA, offset_of_Decoding)
        attack.SingleRound_inject()
        SingleRound_l.append(attack.inject_file_face_TC)
        attack.Decoding_inject()
        Decoding_l.append(attack.inject_file_face_TC)
        attack.BVA_inject()
        BVA_l.append(attack.inject_file_face_TC)
        attack.BVMA_inject()
        BVMA_l.append(attack.inject_file_face_TC)
        attack.ZKP_inject()
        ZKP_l.append(attack.inject_file_face_TC)

    textsize = 15
    plt.rc('font',size=textsize)
    plt.plot(T_list, SingleRound_l, linestyle = '-', color = 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'Single-round')
    plt.plot(T_list, Decoding_l, linestyle = '--', color = 'lightgreen', marker = 'x', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'Decoding')
    plt.plot(T_list, BVA_l, linestyle = '-.', color = 'lightblue', marker = '*', markeredgecolor = 'blue', markersize = 10, markeredgewidth=0.8,  label = 'BVA')
    plt.plot(T_list, BVMA_l, linestyle = '-', color = 'plum', marker = '|', markeredgecolor = 'purple', markersize = 10, markeredgewidth=0.8,  label = 'BVMA')
    plt.plot(T_list, ZKP_l, linestyle = '--', color = 'gray', marker = '.', markeredgecolor = 'k', markersize = 10, markeredgewidth=0.8,  label = 'ZKP')
    plt.yscale('log')
    plt.xlabel("Threshold T")
    plt.ylabel("No. of injected files")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(utils.PLOTS_PATH + '/' + 'TC{}.pdf'.format(dataset_name), bbox_inches = 'tight', dpi = 600)
    plt.show()
