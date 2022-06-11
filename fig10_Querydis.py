import math
import numpy as np
import os
import pickle
import utils
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import attacks
def multiprocess_worker(dataset_name, begin_time, observed_period, target_period, number_queries_per_period, adv_observed_offset, QDGamma):

    """ read data """
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name.lower())), "rb") as f:
        real_size, real_length, offset_of_Decoding = pickle.load(f)
        f.close()

    """ generate queries """
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), 260, adv_observed_offset)
    observed_queries = utils.generate_queries(trend_matrix[:, begin_time:begin_time+observed_period], QDGamma[0], number_queries_per_period)
    target_queries = utils.generate_queries(trend_matrix[:, begin_time+observed_period:begin_time+observed_period+target_period], QDGamma[0], number_queries_per_period)
    attack = attacks.Attack(chosen_kws, observed_queries, target_queries, 1.0, trend_matrix, real_size, real_length, offset_of_Decoding)
    if QDGamma[1]==(int) (offset_of_Decoding/4):
        attack.BVMA_main(QDGamma[0])
    BVMA_accuracy = attack.accuracy  
    attack.BVA_main(QDGamma[1])
    BVA_accuracy = attack.accuracy
    return [BVA_accuracy, BVMA_accuracy]

def plot_figure(dataset_name):
    with open(utils.RESULT_PATH + '/' + 'QueryDis{}.pkl'.format(dataset_name), 'rb') as f:
        (query_distribution, BVA_accuracy, BVMA_accuracy) = pickle.load(f)
        f.close()
    BVA_avr_acc = []
    BVA_min_acc = []
    BVA_max_acc = []
    BVA_accuracy = np.array(BVA_accuracy)
    BVA_avr_acc.append(np.mean(BVA_accuracy[:,0], axis=0))
    BVA_min_acc.append(BVA_avr_acc[0] - np.min(BVA_accuracy[:,0], axis=0))
    BVA_max_acc.append(np.max(BVA_accuracy[:,0], axis=0) - BVA_avr_acc[0])  
    BVA_avr_acc.append(np.mean(BVA_accuracy[:,1], axis=0))
    BVA_min_acc.append(BVA_avr_acc[1] - np.min(BVA_accuracy[:,1], axis=0))
    BVA_max_acc.append(np.max(BVA_accuracy[:,1], axis=0) - BVA_avr_acc[1]) 
 
    textsize = 15
    xticks = np.arange(len(query_distribution))
    fig, ax = plt.subplots()
    ax.bar(xticks, BVA_avr_acc, yerr = [BVA_min_acc, BVA_max_acc], capsize=4, width=0.17, label="BVA", ecolor = 'g',color="lightgreen", edgecolor = "g", linewidth = 0.8)#, hatch = '-'
    ax.bar(xticks+0.17, BVMA_accuracy, width=0.17, label="BVMA", color="r", edgecolor = "white", linewidth = 0.8, hatch = '/')
    ax.set_ylabel('Recovery rate', fontsize = textsize)
    plt.xticks(xticks + 0.17/2, query_distribution)
    ax.legend(fontsize = textsize, loc = 'lower left')
    ax.grid()
    plt.tick_params(labelsize=textsize)
    plt.savefig(utils.PLOTS_PATH +'/' + 'TrendUniformEnron.pdf', bbox_inches = 'tight', dpi=600)
    plt.show()

if __name__ == '__main__':
    
    if not os.path.exists(utils.RESULT_PATH):
        os.makedirs(utils.RESULT_PATH)
    if not os.path.exists(utils.PLOTS_PATH):
        os.makedirs(utils.PLOTS_PATH)
    """ choose dataset """
    dataset_name = 'Enron'
    number_queries_per_period = 1000
    observed_period = 8
    target_period = 10
    adv_observed_offset = 10
    begin_time = 0

    """ read data """
    with open(os.path.join(utils.DATASET_PATH,"{}_doc.pkl".format(dataset_name.lower())), "rb") as f:
        doc = pickle.load(f)
        f.close()
    with open(os.path.join(utils.DATASET_PATH,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    chosen_kws = list(kw_dict.keys())
    with open(os.path.join(utils.DATASET_PATH, "{}_wl_v_off.pkl".format(dataset_name, dataset_name.lower())), "rb") as f:
        real_size, real_length, offset_of_Decoding = pickle.load(f)
        f.close()

    """ experiment """
    exp_times = 1
    query_distribution = ['real-world', 'uniform']
    BVA_gamma_list = []
    minimum_gamma = (int) (len(kw_dict)/2)
    maximum_gamma = (int) (offset_of_Decoding/4)
    minimum_gamma += 1
    BVA_gamma_list.append(minimum_gamma)
    while minimum_gamma<maximum_gamma/2:
        minimum_gamma *= 2
        minimum_gamma += 1
        BVA_gamma_list.append(minimum_gamma)
    BVA_gamma_list.append(maximum_gamma)
    BVMA_accuracy = [0]*len(query_distribution)
    BVA_accuracy =  [[0]*len(query_distribution) for _ in range(len(BVA_gamma_list))]
    total_loop = len(BVA_gamma_list)*2
    pbar = tqdm(total=total_loop)
    for ind in range(len(query_distribution)):
        for ind2 in range(len(BVA_gamma_list)):    
            partial_function = partial(multiprocess_worker, dataset_name, begin_time, observed_period, target_period, number_queries_per_period, adv_observed_offset)
            with Pool(processes=10) as pool:
                for result in pool.map(partial_function, [(query_distribution[ind], BVA_gamma_list[ind2])]*exp_times):
                    BVA_accuracy[ind2][ind] += result[0]
                    BVMA_accuracy[ind] += result[1]
                BVA_accuracy[ind2][ind] /= exp_times
            pbar.update(math.ceil((ind2+1)*(ind+1)/total_loop)) 
        BVMA_accuracy[ind] /= exp_times
    pbar.close()   
    print("BVA_accuracy:{}".format(BVA_accuracy))
    print("BVMA_accuracy:{}".format(BVMA_accuracy))

    """ save result """
    with open(utils.RESULT_PATH + '/' + 'QueryDis{}.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump((query_distribution, BVA_accuracy, BVMA_accuracy), f)
        f.close()

    """ plot figure """
    plot_figure(dataset_name)
    
