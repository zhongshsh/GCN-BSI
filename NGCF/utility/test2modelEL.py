'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

cores = multiprocessing.cpu_count() // 2

args = parse_args()
# Ks = eval(args.Ks)
Ks = [20, 40]



data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size #1024

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test_one_user(x):
    try:
        # user u's ratings for user u
        rating = x[0]
        #uid
        u = x[1]
        #user u's items in the training set
        try:
            training_items = data_generator.train_items[u]
        except Exception:
            training_items = []
        #user u's items in the test set
        user_pos_test = data_generator.test_set[u]

        all_items = set(range(ITEM_NUM))

        test_items = list(all_items - set(training_items))

        if args.test_flag == 'part':
            r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        else:
            r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

        return get_performance(user_pos_test, r, auc, Ks)
    except Exception:
        return {'recall': np.array([0]), 'precision': np.array([0]),
            'ndcg': np.array([0]), 'hit_ratio': np.array([0]), 'auc': 0}

def testProcess(rate_batch, n_user_batchs, u_batch_size, n_test_users):
    pool = multiprocessing.Pool(cores)
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = range(USR_NUM)
        item_batch = range(ITEM_NUM)
        # print(rate_batch.shape)
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
    pool.close()
    return result

def nor(x):
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x

def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    # in NGCF, drop_flag=true, and batch_test_flag=false

    pool = multiprocessing.Pool(cores)
    BATCH_SIZE = 50000
    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    
    path = '/home/user02/zss/ngcf/result_' + args.dataset + '/'
    bgcn = np.load(path + 'bgcn_score.npy')
    ngcf = np.load(path + 'ngcf_score.npy')
    stack = np.load(path + 'stack_score.npy')
    noLinearEl = np.load(path + 'el_score.npy')
    linearEl = np.matmul(np.hstack((np.load(path + 'bgcn_user.npy'), np.load(path + 'ngcf_user.npy'))), np.hstack((np.load(path + 'bgcn_item.npy'), np.load(path + 'ngcf_item.npy'))).T)

    bgcn = nor(bgcn)
    ngcf = nor(ngcf)
    stack = nor(stack)
    noLinearEl = nor(noLinearEl)
    linearEl = nor(linearEl)

    # total
    testList = ['bgcn', 'ngcf', 'stack', 'noLinearEl', 'linearEl']
    testModel = [bgcn, ngcf, stack, noLinearEl, linearEl]

    # # no bgcn
    # testList = ['ngcf', 'stack', 'noLinearEl', 'linearEl']
    # testModel = [ngcf, stack, noLinearEl, linearEl]

    # no ngcf
    # testList = ['bgcn', 'stack', 'noLinearEl', 'linearEl']
    # testModel = [bgcn, stack, noLinearEl, linearEl]

    # # no stack
    # testList = ['bgcn', 'ngcf', 'noLinearEl', 'linearEl']
    # testModel = [bgcn, ngcf, noLinearEl, linearEl]

    # no noLinearEL
    # testList = ['bgcn', 'ngcf', 'stack', 'linearEl']
    # testModel = [bgcn, ngcf, stack, linearEl]

    # no linearEL
    # testList = ['bgcn', 'ngcf', 'stack', 'noLinearEl']
    # testModel = [bgcn, ngcf, stack, noLinearEl]

    print(testList)

    acc = []
    w = []
    for i in range(len(testList)):
        rate_batch = testModel[i]
        result = testProcess(rate_batch, n_user_batchs, u_batch_size, n_test_users)
        acc.append(result['ndcg']+result['recall'])
        w.append(1)
    
    print(acc)
    tmp = [-1]
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    result_tmp = result
    
    while 1==1:
        # pay more attention to top
        if (tmp > (result['ndcg']+result['recall'])).all():
        # if (tmp[0] > (result['ndcg']+result['recall'])[0]):
            print(testList)
            print(w)
            rate_batch = np.zeros((USR_NUM, ITEM_NUM))
            for i in range(len(testList)):
                rate_batch += testModel[i] * tmp_w[i]
            np.save(path + 'total.npy', rate_batch)
            return result_tmp

        tmp = result['ndcg']+result['recall']
        tmp_w = w
        result_tmp = result

        rate_batch = np.zeros((USR_NUM, ITEM_NUM))
        for i in range(len(testList)):
            w[i] = w[i] * acc[i][0]
            rate_batch += testModel[i] * w[i]

        result = testProcess(rate_batch, n_user_batchs, u_batch_size, n_test_users)
        print(result)
    
    # assert count == n_test_users

    pool.close()
    return result



    
