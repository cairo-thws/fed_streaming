"""
MIT License

Copyright (c) 2024 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from .vessal import StreamingSampling
import torch
# import cholesky update lib for low-rank matrix manipulation
import cholupdates

# JITTER FACTOR
CHOLESKY_JITTER = 1e-4

class FedStreamingSampling(StreamingSampling):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(FedStreamingSampling, self).__init__(X, Y, idxs_lb, net, handler, args)


    def streaming_sampler(self, samps, k, early_stop=False, streaming_method='det', \
                        cov_inv_scaling=100, embs="grad_embs"):
        inds = []
        skipped_inds = []
        if embs == "penultimate":
            samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        dim = samps.shape[-1]
        rank = samps.shape[-2]

        # initialize covariance and scale 
        covariance = torch.zeros(dim,dim).cuda() + torch.eye(dim).cuda() * CHOLESKY_JITTER
        covariance_inv = cov_inv_scaling * torch.eye(dim).cuda()
        
        # initialize lower cholesky triangular matrix
        L = torch.linalg.cholesky(covariance)
        
        # convert samples
        samps = torch.tensor(samps)
        samps = samps.cuda()

        for i, u in enumerate(samps):
            if i % 1000 == 0: print(i, len(inds), flush=True)
            if rank > 1: u = torch.Tensor(u).t().cuda()
            else: u = u.view(-1, 1)
            
            # get determinantal contribution (matrix determinant lemma)
            if rank > 1:
                norm = torch.abs(torch.det(u.t() @ covariance_inv @ u))
            else:
                norm = torch.abs(u.t() @ covariance_inv @ u)

            ideal_rate = (k - len(inds))/(len(samps) - (i))
            # just average everything together: \Sigma_t = (t-1)/t * A\{t-1}  + 1/t * x_t x_t^T
            covariance = (i/(i+1))*covariance + (1/(i+1))*(u @ u.t())

            self.zeta = (ideal_rate/(torch.trace(covariance @ covariance_inv))).item()

            # calculate sample-specific probability
            pu = np.abs(self.zeta) * norm
            if np.random.rand() < pu.item():
                inds.append(i)
                if early_stop and len(inds) >= k:
                    break
                # perform cholesky low-rank update
                L = self.cholupdate_seeger(L, u, torch.device("cuda"))
                # calculate inverse of cholseky lower 
                covariance_inv = torch.cholesky_inverse(L)
                # print('compute time cholesky update(sec):', time.time() - start_time_cholesky, flush=True)
            else:
                skipped_inds.append(i)

        return inds, skipped_inds
    

    def cholupdate_seeger(self, L, x, device):
        # convert
        if device.type == 'cuda':
            L_np = np.asarray(L.to("cpu", non_blocking=True))
            x_np = np.asarray(x.to("cpu", non_blocking=True))
            torch.cuda.synchronize()
        else:
            L_np = L.numpy()
            x_np = x.numpy()
        
        # call to efficient low-rank update lib
        cholupdates.rank_1.update(L_np, np.squeeze(x_np, axis=1), check_diag=False, overwrite_L=True, overwrite_v=True,
            method="seeger", impl="cython")
        torch.cuda.synchronize()
        
        # convert
        if device.type == 'cuda':
            ret = torch.from_numpy(L_np).to(device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            ret = L
            
        return ret