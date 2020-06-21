import scipy.sparse as sp
import numpy as np
import lightfm
import seaborn as sns
import matplotlib.pyplot as plt

def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-'*spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif (isinstance(r, float)
              or isinstance(r, np.float32)
              or isinstance(r, np.float64)):
            middle += '| {0:^{1}.5f} '.format(r, spacing-2)
        bottom += '+{}'.format('='*spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


def patk_learning_curve(model, train, test, # eval_train,
                        iterarray, user_features=None,
                        item_features_train=None, 
                        item_features_test=None,
                        k_precision=3, k_recall=100,
                        **fit_params):
    old_epoch = 0
    train_precisions = []
    test_precisions = []
    train_recalls = []
    test_recalls = []
    
    headers = ['Epoch', 'train p@'+str(k_precision), 'test p@'+str(k_precision), 
               'train r@'+str(k_recall), 'test r@'+str(k_recall)]
    print_log(headers, header=True)
    for epoch in iterarray:
        more = epoch - old_epoch
        model.fit_partial(train, user_features=user_features,
                          item_features=item_features_train,
                          epochs=more, **fit_params)
        precision_test = lightfm.evaluation.precision_at_k(model, test, 
                                                           item_features=item_features_test,
                                                           k=k_precision)
        precision_train = lightfm.evaluation.precision_at_k(model, train, 
                                                            item_features=item_features_train,
                                                            k=k_precision)
        
        recall_test = lightfm.evaluation.recall_at_k(model, test, k=k_recall, item_features=item_features_test)
        recall_train = lightfm.evaluation.recall_at_k(model, train, k=k_recall, item_features=item_features_train)

        train_precisions.append(np.mean(precision_test))
        test_precisions.append(np.mean(precision_train))
        train_recalls.append(np.mean(recall_train))
        test_recalls.append(np.mean(recall_test))
        
        row = [epoch, train_precisions[-1], test_precisions[-1], train_recalls[-1], test_recalls[-1]]
        print_log(row)
    return model, train_precisions, test_precisions, train_recalls, test_recalls

import seaborn as sns
sns.set_style('white')

def plot_patk(iterarray, patk,
              title, k=5):
    plt.plot(iterarray, patk);
    plt.title(title, fontsize=20);
    plt.xlabel('Epochs', fontsize=24);
    plt.ylabel('p@{}'.format(k), fontsize=24);
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);

def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
    """Limit interactions df to minimum row and column interactions.

    Parameters
    ----------
    df : DataFrame
        DataFrame which contains a single row for each interaction between
        two entities. Typically, the two entities are a user and an item.
    row_name : str
        Name of column in df which corresponds to the eventual row in the
        interactions matrix.
    col_name : str
        Name of column in df which corresponds to the eventual column in the
        interactions matrix.
    row_min : int
        Minimum number of interactions that the row entity has had with
        distinct column entities.
    col_min : int
        Minimum number of interactions that the column entity has had with
        distinct row entities.
    Returns
    -------
    df : DataFrame
        Thresholded version of the input df. Order of rows is not preserved.

    Examples
    --------

    df looks like:

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004
      1002  |  2002

    thus, row_name = 'user_id', and col_name = 'item_id'

    If we were to set row_min = 2 and col_min = 1, then the returned df would
    look like

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004

    """

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    done = False
    while not done:
        starting_shape = df.shape[0]
        col_counts = df.groupby(row_name)[col_name].count()
        df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
        row_counts = df.groupby(col_name)[row_name].count()
        df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df


def get_df_matrix_mappings(df, row_name, col_name):
    """Map entities in interactions df to row and column indices

    Parameters
    ----------
    df : DataFrame
        Interactions DataFrame.
    row_name : str
        Name of column in df which contains row entities.
    col_name : str
        Name of column in df which contains column entities.

    Returns
    -------
    rid_to_idx : dict
        Maps row ID's to the row index in the eventual interactions matrix.
    idx_to_rid : dict
        Reverse of rid_to_idx. Maps row index to row ID.
    cid_to_idx : dict
        Same as rid_to_idx but for column ID's
    idx_to_cid : dict
    """


    # Create mappings
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_matrix(df, row_name, col_name, filter_in_train=False, filter_in_test=False):
    """Take interactions dataframe and convert to a sparse matrix

    Parameters
    ----------
    df : DataFrame
    row_name : str
    col_name : str

    Returns
    -------
    interactions : sparse csr matrix
    rid_to_idx : dict
    idx_to_rid : dict
    cid_to_idx : dict
    idx_to_cid : dict

    """
    if filter_in_train:
        df = df[df[row_name].isin(set(df[df.dataset=='train'][row_name]))]
    if filter_in_test:
        df = df[df[row_name].isin(set(df[df.dataset=='test'][row_name]))]
        
    rid_to_idx, idx_to_rid,\
        cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,
                                                        row_name,
                                                        col_name)

    def map_ids(row, mapper):
        return mapper[row]
    
    result = {}
    
    for ds in ['train','test']:
        I = df[row_name][df.dataset==ds].apply(map_ids, args=[rid_to_idx]).to_numpy()
        J = df[col_name][df.dataset==ds].apply(map_ids, args=[cid_to_idx]).to_numpy()
        V = df.score[df.dataset==ds].values
        try:
            interactions = sp.coo_matrix((V, (I, J)), 
                                         shape=(df[row_name].nunique(), df[col_name].nunique()), 
                                         dtype=np.float64)
            result[ds] = interactions.tocsr()
        except:
            print(ds)
            raise
    
    return result['train'], result['test'], rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid
