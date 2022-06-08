"""
This code is for Fig. 3.
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib
import os
import utils
import numpy as np

def get_new_queries(kw_dict, number_queries_per_periods, observed_periods, adv_observed_offset):
    _, trend_matrix, _ = utils.generate_keyword_trend_matrix(kw_dict, len(kw_dict), observed_periods, adv_observed_offset)
    queries = utils.generate_queries(trend_matrix, 'trend', number_queries_per_periods)
    ObservedTimes = []
    for i in range(observed_periods):
        ObservedTimes.append(i+1)
    TrendNewQueries = []
    TrendTotalQueries = {}
    for Q in queries:
        new_query_count = 0
        for q in Q:
            if q not in TrendTotalQueries.keys():
                new_query_count += 1
                TrendTotalQueries[q] = 0
        TrendNewQueries.append(new_query_count)
    print(TrendNewQueries)

    queries = utils.generate_queries(trend_matrix, 'uniform', number_queries_per_periods)
    UniformNewQueries = []
    UniformTotalQueries = {}
    for Q in queries:
        new_query_count = 0
        for q in Q:
            if q not in UniformTotalQueries.keys():
                new_query_count += 1
                UniformTotalQueries[q] = 0
        UniformNewQueries.append(new_query_count)
    print(UniformNewQueries)

    return ObservedTimes, TrendNewQueries, UniformNewQueries


from matplotlib.pyplot import MultipleLocator
def plot_figure(ObservedTimes, TrendNewQueries, UniformNewQueries, dataset_name):
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
    matplotlib.rc('font',**font)

    _, ax = plt.subplots()
    plt.plot(ObservedTimes, TrendNewQueries, 'r-', label = 'real-world')
    plt.plot(ObservedTimes, UniformNewQueries, 'g--', label = 'uniform')
    plt.grid() 
    ax.legend(fontsize = 20, loc = 'upper right')
    plt.tick_params(labelsize=20)
    if dataset_name=='Wiki':
        x_major_locator=MultipleLocator(16) 
        ax.set_xlabel('Observed months', fontsize = 20)
    else:   
        x_major_locator=MultipleLocator(4) 
        ax.set_xlabel('Observed weeks', fontsize = 20)
    ax.set_ylabel('New queries', fontsize = 20)
    
    
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('FigureResults\\NewQueries{}.pdf'.format(dataset_name), bbox_inches = 'tight')


if __name__ == '__main__':
    d_id = input("input evaluation dataset: 1. Enron 2. Lucene 3.WikiPedia ")
    dataset_name = ''
    print(d_id)
    exp_times = 10
    number_queries_per_periods = 1000
    observed_periods = 20
    adv_observed_offset = 5

    if d_id=='1':
        dataset_name = 'Enron'
    elif d_id=='2':
        dataset_name = 'Lucene'  
    elif d_id=='3':
        dataset_name = 'Wiki'
        number_queries_per_periods = 5000
        observed_periods = 50
    else:
        raise ValueError('No Selected Dataset!!!')
    
    ObservedTimes = [0]*observed_periods
    TrendNewQueries = [0]*observed_periods
    UniformNewQueries = [0]*observed_periods
    dataset_path = os.path.join(utils.ROOT_DIR, r'Datasets\{}'.format(dataset_name))
    with open(os.path.join(dataset_path,"{}_kws_dict.pkl".format(dataset_name.lower())), "rb") as f:
        kw_dict = pickle.load(f)
        f.close()
    for _ in range(exp_times):
        Ob, Tr, Uni = get_new_queries(kw_dict, number_queries_per_periods, observed_periods, adv_observed_offset)
        ObservedTimes = np.sum([Ob,ObservedTimes], axis=0)
        TrendNewQueries = np.sum([Tr,TrendNewQueries], axis=0)
        UniformNewQueries = np.sum([Uni,UniformNewQueries], axis=0)
    print(np.divide(ObservedTimes,exp_times))
    plot_figure(np.divide(ObservedTimes,exp_times), np.divide(TrendNewQueries,exp_times), np.divide(UniformNewQueries,exp_times), dataset_name)



