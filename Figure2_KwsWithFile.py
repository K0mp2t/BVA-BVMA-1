"""
This code is for Fig. 2.
"""

import os
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib

def get_kws_from_known_doc(doc, chosen_kws, divide_doc_percentage):
    KNOWN_LIST = []
    for i in divide_doc_percentage:
        known_partial_doc = doc[:(int) (len(doc)*i)]
        known_kws = {}
        for kws in known_partial_doc:
            for kw in kws:
                if kw in chosen_kws.keys():
                    known_kws[kw] = 0
        KNOWN_LIST.append(len(known_kws))
    return KNOWN_LIST
def plot_figure(divide_doc_percentage, known_kws_from_enron_dataset):
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
    textsize = 20
    plt.plot(divide_doc_percentage, known_kws_from_enron_dataset, 'lightsalmon', marker = 'o', markeredgecolor = 'red', markersize = 10, markeredgewidth=0.8, label = 'Enron File')
    plt.plot(divide_doc_percentage, known_kws_from_lucene_dataset, 'lightgreen', marker = 's', markeredgecolor = 'green', markersize = 10, markeredgewidth=0.8, label = 'Lucene File')
    plt.xlabel('File leakage percentage', fontsize = textsize)
    plt.ylabel('No. of Kws in Enron', fontsize = textsize)
    plt.tick_params(labelsize = textsize)
    plt.grid()
    plt.legend(fontsize = textsize)
    plt.savefig('FigureResults//KWDistribution.pdf', bbox_inches = 'tight')

if __name__=='__main__':
    enron_dataset_path = './/Datasets//Enron'
    lucene_dataset_path = './/Datasets//Lucene'
    with open(os.path.join(enron_dataset_path,"enron_doc.pkl"), "rb") as f:
        enron_doc = pickle.load(f)
        f.close()
    with open(os.path.join(enron_dataset_path,"enron_kws_dict.pkl"), "rb") as f:
        enron_kw_dict = pickle.load(f)
        f.close()
    with open(os.path.join(lucene_dataset_path,"lucene_doc.pkl"), "rb") as f:
        lucene_doc = pickle.load(f)
        f.close()

    enron_chosen_kws = enron_kw_dict
    random.shuffle(enron_doc)
    random.shuffle(lucene_doc)
    divide_doc_percentage = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1]

    known_kws_from_enron_dataset = get_kws_from_known_doc(enron_doc, enron_chosen_kws, divide_doc_percentage)
    known_kws_from_lucene_dataset = get_kws_from_known_doc(lucene_doc, enron_chosen_kws, divide_doc_percentage)

    plot_figure(divide_doc_percentage, known_kws_from_enron_dataset)
