# encoding: utf-8
"""
Algorithms to synthesize data set: currently differential privacy
"""
import concurrent.futures
import itertools
import os
import time
import warnings
import numpy as np

from itertools import product
from pandas import DataFrame, merge
from scipy.optimize import fsolve

from ds4ml.utils import mutual_information, normalize_distribution, FREQ_NAME

# -----------------------------------------------------------------------------
# Algorithms PrivBayes: Private Data Release via Bayesian Networks


def calculate_sensitivity(n_rows, child, parents, binaries):
    """
    Lemma 4.1 Page 12: Sensitivity function for Bayesian network construction.

    Parameters
    ----------
    n_rows : int
        Number of tuples in dataset
    child : str
        One column name
    parents : tuple
        Parents of child, there may be multiple parents
    binaries : list
        List of binary columns
    """
    if child in binaries or (len(parents) == 1 and parents[0] in binaries):
        a = np.log(n_rows) / n_rows
        b = (1 / n_rows - 1) * np.log(1 - 1 / n_rows)
    else:
        a = (2 / n_rows) * np.log((n_rows + 1) / 2)
        b = (1 - 1 / n_rows) * np.log(1 + 2 / (n_rows - 1))
    return a + b


def usefulness(degree, n_rows, n_cols, epsilon, threshold):
    """
    Lemma 4.8 Page 16: Usefulness measure of each noisy marginal distribution
    """
    theta = n_rows * epsilon / ((n_cols - degree) * (2 ** (degree + 2)))
    return theta - threshold


def calculate_degree(n_rows, n_cols, epsilon):
    """
    Lemma 5.8 Page 16: The largest degree that guarantee theta-usefulness.
    """
    threshold = 5  # threshold of usefulness from Page 17
    default = min(3, int(n_cols / 2))
    args = (n_rows, n_cols, epsilon, threshold)
    warnings.filterwarnings("error")
    try:
        degree = fsolve(usefulness, np.array(int(n_cols / 2)), args=args)[0]
        degree = int(np.ceil(degree))
    except RuntimeWarning:
        warnings.warn('Degree of bayesian network is not properly computed!')
        degree = default
    if degree < 1 or degree > n_cols:
        degree = default
    return degree


def greedy_bayes(dataset: DataFrame, epsilon, degree=None, retains=None):
    """
    Algorithm 4, Page 20: Construct bayesian network by greedy algorithm.

    Parameters
    ----------
    dataset : DataFrame
        Encoded dataset
    epsilon : float
        Parameter of differential privacy
    degree : int
        Degree of bayesian network. If null, calculate it automatically.
    retains : list
        The columns to retain
    """
    dataset = dataset.astype(str, copy=False)
    n_rows, n_cols = dataset.shape
    retains = retains or []
    if not degree:
        degree = calculate_degree(n_rows, n_cols, epsilon)

    # list to store the structure of Bayesian Network
    # its element: [child, [parent]]
    network = []
    mi_cache = {}  # cache for mutual information calculation results

    pairwise_mi = {}
    cpu_assigned = int(os.cpu_count() * 0.5)
    tasks = itertools.combinations(dataset.columns, 2)

    def _cache_key(_child, _parents):
        return f'{"#D4@5S#".join(_parents)}@S5#4D@{_child}'

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_assigned) as executor:
        futures = {
            executor.submit(mutual_information, dataset[t], dataset[s]): (t, s)
            for t, s in tasks
        }
        for f in concurrent.futures.as_completed(futures):
            col = futures[f]  # (child, parent)
            try:
                data = f.result()
            except Exception as e:
                print(f'can\'t calculate mutual information in {col}, because {e}.')
            else:
                pairwise_mi[col] = data
                p_key = _cache_key(col[0], [col[1]])
                mi_cache[p_key] = data
                # print(p_key, data)

    # filter outer columns whose mutual information < 0.2 (independent columns),
    # which are not strong enough to be nodes in Bayesian Network.
    ind_columns = []
    for col in dataset.columns:
        if all(pairwise_mi[k] < 0.2 for k in pairwise_mi if k[0] == col or k[1] == col):
            network.append((col, []))
            ind_columns.append(col)
    print('ind_columns:', ind_columns)
    print('\t', network)

    # filter outer columns whose mutual information > 0.8 (dependent columns),
    # they are nodes in Bayesian Network, and degree is 1 (highly correlated).
    dep_columns = []
    for col in dataset.columns:
        pairs = [k for k in pairwise_mi if k[0] == col and pairwise_mi[k] > 0.8]
        if len(pairs) > 0:
            if pairs[0][0] not in dep_columns:
                dep_columns.append(pairs[0][0])
                network.append((pairs[0][0], []))
            for k in pairs:
                if k[1] not in dep_columns:
                    dep_columns.append(k[1])
                    network.append((k[1], [k[0]]))
    print('dep_columns:', dep_columns)
    print('\t', network)

    # mapping from column name to is_binary, because sensitivity is different
    # for binary or non-binary column
    binaries = [col for col in dataset if dataset[col].unique().size <= 2]
    more_retains = False
    # columns: a set that contains all attributes whose parent sets has been set
    columns = dep_columns
    retains = [c for c in retains if c not in dep_columns and c not in ind_columns]

    # columns to be determined to add to Bayesian Network
    undetermined = list(set(dataset.columns) - set(dep_columns) - set(ind_columns))
    if len(retains) == 0:
        root_col = np.random.choice(undetermined)
        # root_col = max(undetermined, key=)
        # root_col = np.random.choice(dataset.columns)
    elif len(retains) == 1:
        root_col = retains[0]
    else:
        root_col = np.random.choice(retains)
        more_retains = True

    columns.append(root_col)
    network.append((root_col, []))
    if more_retains:
        left_cols = set(retains)
    else:
        left_cols = set(undetermined)
        # left_cols = set(dataset.columns)
    # print(root_col in left_cols)
    # if root_col in left_cols:
    left_cols.remove(root_col)

    def _candidate_pairs(paras):
        """
        Return attribute-parents pairs, and their mutual information.
        """
        from itertools import combinations
        _child, _columns, _n_parents, _index, _dataset = paras
        _aps = []
        _mis = []

        if _index + _n_parents - 1 < len(_columns):
            for _parents in combinations(_columns[_index + 1:], _n_parents - 1):
                _parents = list(_parents)
                _parents.append(_columns[_index])
                _aps.append((_child, _parents))
                # combination of column names to prevent duplicate calculation
                # of mutual information
                # _key = f'{"#D4@5S#".join(_parents)}@S5#4D@{_child}'
                _key = _cache_key(_child, _parents)
                if _key in mi_cache:
                    _mi = mi_cache[_key]
                else:
                    _mi = mutual_information(_dataset[_child], _dataset[_parents])
                    mi_cache[_key] = _mi
                _mis.append(_mi)
        return _aps, _mis

    # print('-x' * 10)
    # print('left_cols:', left_cols)
    # print('remained_columns:', remained_columns)
    # print(more_retains)
    # print(root_col)
    while len(left_cols) > 0:
        # ap: attribute-parent (AP) pair is a tuple. It is a node in bayesian
        # network, e.g. ('education', ['relationship']), there may be multiple
        # parents, depends on k (degree of bayesian network).
        aps = []
        # mi: mutual information (MI) of two features
        mis = []
        n_parents = min(len(columns), degree)
        # n_parents = min(len(columns), degree)
        # calculate the candidate set of attribute-parent pair
        # print(len(columns) - n_parents + 1)
        tasks = [(child, columns, n_parents, index, dataset) for child, index in
                 product(left_cols, range(len(columns) - n_parents + 1))]
        # TODO: should use thread pool for large data set?
        # print('tasks:', len(tasks))
        # print([(child, columns, n_parents, index) for child, index in
        #          product(left_cols, range(len(columns) - n_parents + 1))])
        candidates = list(map(_candidate_pairs, tasks))
        # print('candidates:', candidates)
        for ap, mi in candidates:
            aps += ap
            mis += mi
        # find next child node in bayesian networks according to the biggest
        # mutual information or exponential mechanism
        if epsilon:
            index = sampling_pair(mis, aps, binaries, n_rows, n_cols, epsilon)
        else:
            index = mis.index(max(mis))
        network.append(aps[index])
        next_col = aps[index][0]
        columns.append(next_col)
        left_cols.remove(next_col)
        if len(left_cols) == 0 and more_retains:
            left_cols = set(undetermined) - set(retains)
            # left_cols = set(dataset.columns) - set(retains)
            more_retains = False
    return network


def sampling_pair(mis, aps, binaries, n_rows, n_cols, epsilon):
    """
    Page 6 and 12: Sampling an attribute-parent pair from candidates by
    exponential mechanism.
    """
    deltas = []
    for child, parents in aps:
        sensitivity = calculate_sensitivity(n_rows, child, parents, binaries)
        delta = (n_cols - 1) * sensitivity / epsilon
        deltas.append(delta)
    prs = np.array(mis) / (2 * np.array(deltas))
    prs = np.exp(prs)
    prs = normalize_distribution(prs)
    return np.random.choice(list(range(len(mis))), p=prs)


def noisy_distributions(dataset, columns, epsilon):
    """
    Generate differentially private distribution by adding Laplace noise
    Algorithm 1 Page 9: parameters (scale, size) of Laplace distribution
    """
    data = dataset.copy()[columns]
    data[FREQ_NAME] = 1
    freq = data.groupby(columns).sum()
    freq.reset_index(inplace=True)

    iters = [range(int(dataset[col].max()) + 1) for col in columns]
    domain = DataFrame(columns=columns, data=list(product(*iters)))
    # freq: the complete probability distribution
    freq = merge(domain, freq, how='left')
    freq.fillna(0, inplace=True)

    n_rows, n_cols = dataset.shape
    scale = 2 * (n_cols - (len(columns) - 1)) / (n_rows * epsilon)
    if epsilon:
        noises = np.random.laplace(0, scale=scale, size=freq.shape[0])
        freq[FREQ_NAME] += noises
        freq.loc[freq[FREQ_NAME] < 0, FREQ_NAME] = 0
    return freq


def noisy_conditionals(network, dataset, epsilon):
    """
    Algorithm 1, Page 9: noisy conditional distribution probability
    """
    cond_prs = {}  # conditional probability distributions

    # test another method to get cond_prs
    # _c, _p = network[0]
    # _k = noisy_distributions(dataset, [_c] + _p, epsilon)
    # _r = _p[0]
    # _r_pr = _k[[_r, 'freq']].groupby(_r).sum()['freq']
    # print('another cond_prs:')
    # print(normalize_distribution(_r_pr).tolist())

    # print('----- network ---')

    for child, parents in network:
        print(child, parents)
        freq = noisy_distributions(dataset, parents + [child], epsilon)
        # independent columns or parent of dependent columns
        # # for backward compatibility
        if len(parents) == 0 or (len(parents) == 1 and parents[0] not in cond_prs):
            root = child if len(parents) == 0 else parents[0]
            root_prs = freq[[root, FREQ_NAME]].groupby(root).sum()[FREQ_NAME]
            cond_prs[root] = normalize_distribution(root_prs).tolist()
            # print('no parents:', root)
            if len(parents) == 0:
                continue

        freq = DataFrame(freq[parents + [child, FREQ_NAME]]
                         .groupby(parents + [child]).sum())
        cond_prs[child] = {}
        # print('with parent:', child, parents)
        # print(freq.head(2).to_string())
        if len(parents) == 1:
            for parent in freq.index.levels[0]:
                prs = normalize_distribution(freq.loc[parent][FREQ_NAME]).tolist()
                cond_prs[child][str([parent])] = prs
        else:
            for parent in product(*freq.index.levels[:-1]):
                prs = normalize_distribution(freq.loc[parent][FREQ_NAME]).tolist()
                cond_prs[child][str(list(parent))] = prs

    # # distribution of one or more root node(s) in bayesian network
    # root = network[0][1][0]
    # # attributes [1, k]
    # kattr = [root]
    # for child, _ in network[:len(network[-1][1])]:
    #     kattr.append(child)
    #
    # kfreq = noisy_distributions(dataset, kattr, epsilon)
    # root_prs = kfreq[[root, 'freq']].groupby(root).sum()['freq']
    # cond_prs[root] = normalize_distribution(root_prs).tolist()
    #
    # # distributions of other child node(s) in bayesian network
    # net_idx = 0
    # for child, parents in network:
    #     cond_prs[child] = {}
    #     if net_idx < len(network[-1][1]):
    #         freq = kfreq.copy().loc[:, parents + [child, 'freq']]
    #     else:
    #         freq = noisy_distributions(dataset, parents + [child], epsilon)
    #     freq = DataFrame(freq[parents + [child, 'freq']]
    #                      .groupby(parents + [child]).sum())
    #     if len(parents) == 1:
    #         for parent in freq.index.levels[0]:
    #             prs = normalize_distribution(freq.loc[parent]['freq']).tolist()
    #             cond_prs[child][str([parent])] = prs
    #     else:
    #         for parent in product(*freq.index.levels[:-1]):
    #             prs = normalize_distribution(freq.loc[parent]['freq']).tolist()
    #             cond_prs[child][str(list(parent))] = prs
    #     net_idx = net_idx + 1
    return cond_prs
