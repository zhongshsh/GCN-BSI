import numpy as np


def nor(x):
    Min = np.min(x)
    Max = np.max(x)
    x = (x - Min) / (Max - Min)
    return x


dataset = "steamgame"
path = "/home/user02/zss/ngcf/result_" + dataset + "/"
bgcn = np.load(path + "bgcn_score.npy")
ngcf = np.load(path + "ngcf_score.npy")
stack = np.load(path + "stack_score.npy")
noLinearEl = np.load(path + "el_score.npy")
linearEl = np.matmul(
    np.hstack((np.load(path + "bgcn_user.npy"), np.load(path + "ngcf_user.npy"))),
    np.hstack((np.load(path + "bgcn_item.npy"), np.load(path + "ngcf_item.npy"))).T,
)

bgcn = nor(bgcn)
ngcf = nor(ngcf)
stack = nor(stack)
noLinearEl = nor(noLinearEl)
linearEl = nor(linearEl)

models = [bgcn, ngcf, stack, noLinearEl, linearEl]
w = [
    0.0012914509218483541,
    0.06407250098637728,
    0.19467171757193327,
    0.008041256332115302,
    0.8523580405002879,
]
res = models[0] * w[0]
for i in range(len(models) - 1):
    i += 1
    res += models[i] * w[i]

np.save(path + "total.npy", res)
