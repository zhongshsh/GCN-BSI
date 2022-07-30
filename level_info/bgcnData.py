# with open('/data/bgcn/bgcn-scores/mix-weight.txt', 'r') as f:
#     bmat = f.read().strip().split('\n')

# for i in range(len(bmat)):
#     bmat[i] = bmat[i].split('\t')

# bmat = tf.convert_to_tensor(bmat)
# print(bmat.shape)

# import tensorflow as tf
import numpy as np
import torch
import scipy.sparse as sp
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 3, 2"
torch.cuda.set_device(0)
num_user = 18528
num_bundle = 22864
num_item = 123628
ub = torch.from_numpy(np.load("bgcn_ub.npy"))
ub = torch.tensor(ub, dtype=torch.float32)
print("ub: ", str(ub.shape))
bi = torch.from_numpy(np.load("bgcn_bi.npy"))
bi = torch.tensor(bi, dtype=torch.float32)
print("bi: ", str(bi.shape))
scores = torch.einsum("ik, kj -> ij", ub, bi).numpy()
print("scores: ", str(scores.shape))
np.save("bgcn_scores.npy", scores)

# ub data
# with open('/data/bgcn/bgcn-scores/mix-weight.txt', 'r') as f:
#     ub = list(map(lambda s: tuple(eval(i) for i in s[:-1].split('\t')), f.read().strip().split('\t\n')))
# indices = np.vstack((num_user, num_bundle))
# ub = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(ub),
#                                           torch.Size((num_user, num_bundle)))

# ub = np.array(ub)
# print("ub data: ", ub.shape) #  (18528, 22864)
# np.save('bgcn_ub.npy',ub)


# # bi data
# def to_tensor(graph):
#     graph = graph.tocoo()
#     values = graph.data
#     indices = np.vstack((graph.row, graph.col))
#     graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
#                                           torch.Size(graph.shape))
#     return graph

# def laplace_transform(graph):
#     rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
#     colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
#     graph = rowsum_sqrt @ graph @ colsum_sqrt
#     return graph

# with open('/data/prj2020/zss/bundle_info/data/NetEase/bundle_item.txt', 'r') as f:
#     b_i_pairs =  list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
# indice = np.array(b_i_pairs, dtype=np.int32)
# values = np.ones(len(b_i_pairs), dtype=np.float32)
# b_i_pairs = None
# bi = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(num_bundle, num_item)).tocsr()
# # bi = to_tensor(laplace_transform(bi)).to_dense()
# bi = to_tensor(bi).to_dense()
# bi = np.array(bi)
# print("bi data: ", bi.shape)
# np.save('bgcn_bi.npy',bi)


# einsum pytorch
# ub = None
# bi = None
# print("result data:", result.shape)

# saver = tf.train.Saver({
#     "ui": result
# })


# from tensorflow.python.tools import inspect_checkpoint as ickpt
# result = ickpt.print_tensors_in_checkpoint_file("bgcnmodel.ckpt", tensor_name="result", all_tensors=False)


# # ui data
# def to_tensor(graph):
#     graph = graph.tocoo()
#     values = graph.data
#     indices = np.vstack((graph.row, graph.col))
#     graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values),
#                                           torch.Size(graph.shape))
#     return graph

# def laplace_transform(graph):
#     rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
#     colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
#     graph = rowsum_sqrt @ graph @ colsum_sqrt
#     return graph

# num_user = 18528
# num_item = 30001
# with open('/data/prj2020/zss/data/newNetease/user_item.txt', 'r') as f:
#     b_i_pairs =  list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
# indice = np.array(b_i_pairs, dtype=np.int32)
# values = np.ones(len(b_i_pairs), dtype=np.float32)
# b_i_pairs = None
# bi = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(num_user, num_item)).tocsr()
# # bi = to_tensor(laplace_transform(bi)).to_dense()
# bi = to_tensor(bi).to_dense()
# bi = np.array(bi)
# print("ui data: ", bi.shape)
# np.save('./npyData/bgcn_ui.npy',bi)
