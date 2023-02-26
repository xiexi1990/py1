import scipy.io
import torch
import numpy as np
from scipy.sparse.coo import coo_matrix
from pathlib import Path
import time

def scipy2torch(m : coo_matrix, device : str) -> torch.tensor:
    values = m.data
    indices = np.vstack((m.row, m.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)

    return torch.sparse_coo_tensor(i, v, torch.Size(m.shape), device=torch.device(device), dtype=torch.float32).to_sparse_csr()

# fileset = Path('matrices_tapa').rglob('*.mtx')
# for file in fileset:
#     m = scipy.io.mmread(file)
#     x = np.zeros(m.shape[1])
#     for i in range(m.shape[1]):
#         x[i] = 1.0 * (i + 1)
#     tm = scipy2torch(m, 'cuda')
#     tx = torch.tensor(x, device='cuda', dtype=torch.float32)
#     print('warm up ...')
#     for _ in range(1):
#         tm.mv(tx)
#   #  break


#fileset = Path('matrices_tapa').rglob('*.mtx')
# for file in fileset:
#     print(file.stem)
#     m = scipy.io.mmread(file)
#     x = np.zeros(m.shape[1])
#     for i in range(m.shape[1]):
#         x[i] = 1.0 * (i + 1)
#     tm = scipy2torch(m, 'cuda')
#     tx = torch.tensor(x, device='cuda', dtype=torch.float32)
#     print('start mv ...')
#     tic = time.time()
#     for _ in range(1000):
#         tm.mv(tx)
#     print(time.time() - tic)

#fileset = Path('matrices_tapa').rglob('*.mtx')
# for file in fileset:
#     print(file.stem)
#     m = scipy.io.mmread(file)
#     x = np.zeros(m.shape[1])
#     for i in range(m.shape[1]):
#         x[i] = 1.0 * (i + 1)
#     tm = scipy2torch(m, 'cuda')
#     tx = torch.tensor(x, device='cuda', dtype=torch.float32)
#     print('start mv ...')
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     for _ in range(10000):
#         tm.mv(tx)
#     end.record()
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))

# fileset = Path('matrices_tapa').rglob('*.mtx')
# for file in fileset:
#     print(file.stem)
#     m = scipy.io.mmread(file)
#     A = np.ones([m.shape[1], 1000])
#
#     tm = scipy2torch(m, 'cuda')
#     tA = torch.tensor(A, device='cuda', dtype=torch.float32)
#     print('start mv ...')
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     for _ in range(1):
#         tm.matmul(tA)
#     end.record()
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))

fileset = Path('matrices_tapa').rglob('*.mtx')
for file in fileset:
    print(file.stem)
    m = scipy.io.mmread(file)
    A = np.ones([m.shape[1], 1000])
    B = np.ones([m.shape[0], 1000])

    ALPHA = 0.85;
    BETA = -2.06;

    tm = scipy2torch(m, 'cuda')
    tA = torch.tensor(A, device='cuda', dtype=torch.float32)
    tB = torch.tensor(B, device='cuda', dtype=torch.float32)
    print('start mv ...')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        Y = ALPHA * tm.matmul(tA) + BETA * tB
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    print(Y.shape)

fileset = Path('matrices_tapa').rglob('*.mtx')
for file in fileset:
    print(file.stem)
    m = scipy.io.mmread(file)
    A = np.ones([m.shape[1], 1000])
    B = np.ones([m.shape[0], 1000])

    ALPHA = 0.85;
    BETA = -2.06;

    tm = scipy2torch(m, 'cuda')
    tA = torch.tensor(A, device='cuda', dtype=torch.float32)
    tB = torch.tensor(B, device='cuda', dtype=torch.float32)
    print('start mv ...')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1):
        Y = ALPHA * tm.matmul(tA) + BETA * tB
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    print(Y.shape)