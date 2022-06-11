import numpy as np

DATASET_PATH = 'Datasets'
RESULT_PATH = 'Results'
PLOTS_PATH = 'Plots'

def compute_decoding_offset(observed_response_size, threshold):          
    """compute offset of decoding"""
    divided_list = {}
    for i in observed_response_size.keys():
        for j in observed_response_size.keys():
            temp_size_minus = abs(observed_response_size[i] - observed_response_size[j])
            if temp_size_minus!=0:
                divided_list[temp_size_minus] = 0

    
    #with open('divide_list.pkl', 'rb') as f:
    #    divided_list = pickle.load(f)
    #    f.close()
    #print(len(divided_list))
    offset = 0
    off_bound = 10**7
    for injection_offset in range(2, off_bound):
        #print(injection_offset)
        if injection_offset%10000==0:
            print(injection_offset)
        flag = True
        satisfied_size_minus = 0
        for divisor in divided_list.keys():
            if divisor % injection_offset == 0:
                flag = False
                break
            else:
                satisfied_size_minus += 1
                if satisfied_size_minus>=threshold:
                    break     
        if flag:
            offset = injection_offset
            break       
    if offset == 0:
        print("non adaptive offset!!!")
        offset = off_bound
    print("offset: {}".format(offset))
    return offset

def generate_keyword_trend_matrix(kw_dict, n_kw, n_weeks, offset):
    """
    generate keywords and their queries' trend matrix(row: kws; colume: weeks)
    @ param: kw_dict, n_kw, n_weeks, adv. known weeks offset
    @ return: chosen_kws, trend_matrix, offset_trend_matrix
    """
    
    # n_kw number of keywords
    keywords = list(kw_dict.keys())
    permutation = np.random.permutation(len(keywords))
    #chosen_kws = [keywords[idx] for idx in range(n_kw)]
    chosen_kws = [keywords[idx] for idx in permutation[: n_kw]]
    #print(chosen_kws)
    # n_weeks trend
    trend_matrix_norm = np.array([[float(kw_dict[kw]['trend'][i]) for i in range(len(kw_dict[kw]['trend']))] for kw in chosen_kws])
    #print((trend_matrix_norm[:, 1]))
    for i_col in range(trend_matrix_norm.shape[1]):
        if sum(trend_matrix_norm[:, i_col]) == 0:
            print("The {d}th column of the trend matrix adds up to zero, making it uniform!")
            trend_matrix_norm[:, i_col] = 1 / n_kw
        else:
            trend_matrix_norm[:, i_col] =  trend_matrix_norm[:, i_col] / (float) (sum(trend_matrix_norm[:, i_col]))
    #print((trend_matrix_norm[:, 1]))
    #print(max(trend_matrix_norm[:, 1]))
    #print(trend_matrix_norm[:, 1].index(max(trend_matrix_norm[:, 1])))
    return chosen_kws, trend_matrix_norm[:, -n_weeks:], trend_matrix_norm[:, -n_weeks:] if offset ==0 else trend_matrix_norm[:, -offset-n_weeks: -offset]


def generate_queries(trend_matrix_norm, q_mode, n_qr):
    """
    generate queries from different query modes
    @ param: trend_matrix, q_mode = ['trend', 'uniform']
    @ return: queries matrix(each week)(each kw_id in the chosen_kws)
    """
    queries = []
    n_kw, n_weeks = trend_matrix_norm.shape
    if q_mode == 'real-world':
        for i_week in range(n_weeks):
            #n_qr_i_week = np.random.poisson(n_qr)
            n_qr_i_week = n_qr
            queries_i_week = list(np.random.choice(list(range(n_kw)), n_qr_i_week, p = trend_matrix_norm[:, i_week]))
            queries.append(queries_i_week)

    elif q_mode == 'uniform':
        for i_week in range(n_weeks):
            queries_i_week = list(np.random.choice(list(range(n_kw)), n_qr))
            queries.append(queries_i_week)
    else:
        raise ValueError("Query params not recognized")        
    return queries

def get_kws_size_and_length(doc, chosen_kws):
    """
    get all keywords response size and response length.
    """
    real_size = {}
    real_length = {}
    kw_to_id = {}
    for kw_id in range(len(chosen_kws)):
        real_size[kw_id] = 0
        real_length[kw_id] = 0
        kw_to_id[chosen_kws[kw_id]] = kw_id
    for _, cli_doc_kws in enumerate(doc):
        flag = [False]*len(chosen_kws) 
        for doc_kws in cli_doc_kws:
            if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                real_length[kw_to_id[doc_kws]] += 1
                flag[kw_to_id[doc_kws]] = True
    return real_size, real_length

def get_kws_size_and_length_after_padding(doc, chosen_kws, x):
    """
    get all keywords response size and response length.
    """
    real_size = {}
    real_length = {}
    kw_to_id = {}
    for kw_id in range(len(chosen_kws)):
        real_size[kw_id] = 0
        real_length[kw_id] = 0
        kw_to_id[chosen_kws[kw_id]] = kw_id
    for _, cli_doc_kws in enumerate(doc):
        flag = [False]*len(chosen_kws) 
        for doc_kws in cli_doc_kws:
            if doc_kws in kw_to_id.keys() and not flag[kw_to_id[doc_kws]]:
                real_size[kw_to_id[doc_kws]] += len(cli_doc_kws)
                real_length[kw_to_id[doc_kws]] += 1
                flag[kw_to_id[doc_kws]] = True
    # padding
    for k in real_length.keys():
        m = x
        while real_length[k]>m:
            m *= x  
        real_size[k] = real_size[k] + m - real_length[k]
    return real_size, real_length
