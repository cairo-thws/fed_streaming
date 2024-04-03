"""
MIT License

Copyright (c) 2023 Manuel Roeder

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

import time
import numpy as np
import os
import torch
import cholupdates
import statistics
import matplotlib.pyplot as plt
import scipy.stats as stats

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#DIM = 1024
RUNS = 1
NUM_SAMPLES = 1500

CPU_PRECISION = 1e-6
GPU_PRECISION = 1e-12
CHOL_RECALIBRATION_NR = 3

def main():
    #np.random.seed(1)
    #X = np.random.normal(size=(100,10))
    #V = np.dot(X.transpose(),X)
    #Calculate the upper Cholesky factor, R
    #R = np.linalg.cholesky(V).transpose()

    #Create a random update vector, u
    #u = np.random.normal(size=R.shape[0])

    #Calculate the updated positive definite matrix, V1, and its Cholesky factor, R1
    #V1 = V + np.outer(u,u)
    #R1 = np.linalg.cholesky(V1).transpose()
    
    # Set the random seed for reproducibility
    
    torch.manual_seed(1)
    
    dims = [256, 1024, 2048]
    
    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")
    precision = GPU_PRECISION if device.type == "cuda" else CPU_PRECISION
    
    # set the basic properties
    ax = plt.subplot(111)
    ax.set_xlabel('Number of Updates')
    ax.set_ylabel('Relative Error')
    #ax.set_title('Experiment: Reconstruction Error')
    
    name = "Paired"
    cmap = plt.colormaps[name]  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    ax.set_prop_cycle(color=colors)
    
    for dim in dims:
        times_std = list()
        times_chol = list()
        times_wood = list()
        cholesky_precision_good = 0
        woodbury_precision_good = 0
        total_updates = 0
        precision_tracker_chol_abs = list()
        precision_tracker_wood_abs = list()
        precision_tracker_chol_rel = list()
        precision_tracker_wood_rel = list()
        precision_tracker_update_rel = list()
        precision_tracker_update_abs = list()
        
        for run in range(RUNS):
            # Generate random data using PyTorch
            X = torch.randn(dim, dim).to(device=device, non_blocking=True)
            V = torch.mm(X, X.t()).to(device=device, non_blocking=True)
            V += 1e-6 * torch.eye(dim).to(device, non_blocking=True)
            
            V1 = V.detach().clone()
            V_inv_woodbury = torch.inverse(V1) #torch.eye(DIM).cuda()*100 #torch.inverse(V1)#1 * torch.eye(DIM).cuda()#torch.inverse(V1)
            V2 = V.detach().clone()
            #V3 = V.clone()
            R1 = torch.linalg.cholesky(V2)
            torch.cuda.synchronize()
            
            print("Starting...")
            print("%6s\t%10s\t%10s" % ("update", "Cholesky", "Woodbury"))
            for update in range(NUM_SAMPLES):
                # Create a random rank1-update vector, u
                u = torch.randn(V.shape[0]).to(device=device, non_blocking=True)
                torch.cuda.synchronize()
                
                #The following is equivalent to the above
                #V = V + torch.eye(DIM).cuda() * 1e-4
                start_time_cholesky = time.time()
                #R1 = update_rank_one(R1, u, x_copy=False)
                R1 = cholupdate_seeger(R1, u, device)
                #R1 = torch.linalg.cholesky(V_inv_old, upper=True)
                #R1 = cholupdate_torch(R1, u, '+')
                V_inv_chol = torch.cholesky_inverse(R1)
                torch.cuda.synchronize()
                time_cholesky = time.time() - start_time_cholesky
                times_chol.append(time_cholesky)
                #print('compute time cholesky update and inverse(sec):', time_cholesky, flush=True)
                #target = torch.linalg.cholesky(V + torch.outer(u, u), upper=True)
                #print( (R1 - target)**2 < 1e-14)
                #print(torch.allclose(R1, target))
                
                start_time_std = time.time()
                # Calculating V1 and R1 using PyTorch
                V1 = V1 + torch.outer(u, u)
                V_inv_std = torch.inverse(V1)
                torch.cuda.synchronize()
                #R1 = torch.linalg.cholesky(V1, upper=True)
                time_std = time.time() - start_time_std
                times_std.append(time_std)
                #print('compute time std update and inverse(sec):', time_std, flush=True)
                
                
                start_time_woodbury = time.time()
                u_ext = torch.unsqueeze_copy(u, dim=1).to(device)
                # woodbury update to covariance_inv
                inner_inv = torch.inverse(torch.eye(1).to(device) + u_ext.t() @ V_inv_woodbury @ u_ext)
                inner_inv = inf_replace(inner_inv)
                V_inv_woodbury = V_inv_woodbury - V_inv_woodbury @ u_ext @ inner_inv @ u_ext.t() @ V_inv_woodbury
                time_woodbury = time.time() - start_time_woodbury
                times_wood.append(time_woodbury)
                #print('compute time woodbury update and inverse(sec):', time_woodbury, flush=True)
                
                '''
                chol_prec, wood_prec = evaluate_inverse_precision(V_inv_std, V_inv_chol, V_inv_woodbury, precicion=precision)
                if chol_prec:
                    cholesky_precision_good += 1
                if wood_prec:
                    woodbury_precision_good += 1
                '''
                
                if (update+1) % 1 == 0:
                    (method1_abs, method2_abs, method1_rel, method2_rel) = check_convergence(V_inv_std, V_inv_chol, V_inv_woodbury, precision)
                    print("%6d\t%10.4e\t%10.4e" % (update+1, method1_abs, method2_abs))
                    precision_tracker_chol_abs.append(method1_abs)
                    precision_tracker_wood_abs.append(method2_abs)
                    precision_tracker_update_abs.append(update)
                    if (update+1) % 10 == 0:
                        precision_tracker_chol_rel.append(method1_rel)
                        precision_tracker_wood_rel.append(method2_rel)
                        precision_tracker_update_rel.append(update)
                    
                total_updates += 1
                
            print('-'*80)
            print('Run finished:', run, flush=True)
        
        ax.plot(np.array(precision_tracker_update_rel), np.array(precision_tracker_chol_rel), label="$\mathbf{A}_{cho}, d=$" + str(dim), lw=2,)
        ax.plot(np.array(precision_tracker_update_rel), np.array(precision_tracker_wood_rel), label="$\mathbf{A}_{wbf}, d=$" + str(dim), lw=2, linestyle='--')
        
        
    #ax.set_prop_cycle(color=['red', 'green'])
    
    # confidence
    #mean1, lower1, upper1 = mean_confidence_interval(precision_tracker_chol_rel)
    #mean2, lower2, upper2 = mean_confidence_interval(precision_tracker_wood_rel)
    # Calculate the cumulative sum of errors for each method
    #cumulative_errors_chol = torch.cumsum(torch.tensor(precision_tracker_chol_abs), dim=0)
    #cumulative_errors_wood = torch.cumsum(torch.tensor(precision_tracker_wood_abs), dim=0)
    #ax.plot(np.array(precision_tracker_update_abs), np.array(cumulative_errors_chol), label="chol_cumsum")
    #ax.plot(np.array(precision_tracker_update_rel), np.array(precision_tracker_chol_rel), label="$\mathbf{A}_{cho} - ours $", lw=2)
    #plt.fill_between(precision_tracker_update, lower1, upper1, color='blue', alpha=0.3)
    #ax.plot(np.array(precision_tracker_update_abs), np.array(cumulative_errors_wood), label="wood_cumsum")
    #ax.plot(np.array(precision_tracker_update_rel), np.array(precision_tracker_wood_rel), label="$\mathbf{A}_{wbf}$", lw=2)
    #plt.fill_between(precision_tracker_update, lower2, upper2, color='red', alpha=0.3)
    #plt.plot(precision_tracker_update, mean2, color='red', label='List 2')
    
    
    
    leg = plt.legend(loc='best', fontsize=8)
    
    #plt.xlabel("Number of Updates")
    #plt.ylabel("Relative Error")
    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    #ax.grid('on')
    #plt.ylabel("Cumulative Error")
    
    # scaling
    plt.yscale('log')
    #plt.grid(True)
    
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)

    # tweak the title
    ttl = ax.title
    ttl.set_weight('bold')  
    
    plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    print("Done")
    
    print('avg compute time woodbury update and inverse(sec):', statistics.fmean(times_wood), flush=True)
    print('avg compute time std update and inverse(sec):', statistics.fmean(times_std), flush=True)

    print('avg compute time chol update and inverse(sec):', statistics.fmean(times_chol), flush=True)
    print('total low-rank updates:', total_updates, flush=True)

    print('cholesky updates holding precision:', cholesky_precision_good, flush=True)
    print('woodbury updates holding precision:', woodbury_precision_good, flush=True)
    

def inf_replace(mat):
        mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
        return mat
    
    
def check_convergence(in_1, in_2, in_3, tol):
        diff_norm_1 = torch.abs((in_2 - in_1))
        diff_norm_2 = torch.abs((in_3 - in_1))
        diff_rel_1 = torch.where(in_1 != 0, diff_norm_1 / torch.abs(in_1), torch.zeros_like(in_1))
        diff_rel_2 = torch.where(in_1 != 0, diff_norm_2 / torch.abs(in_1), torch.zeros_like(in_1))
        #diff_rel_1 = diff_norm_1 / torch.abs(in_2)
        #diff_rel_2 = diff_norm_2 / torch.abs(in_3)
        return (torch.mean(diff_norm_1).item(), torch.mean(diff_norm_2).item(), torch.mean(diff_rel_1).item(), torch.mean(diff_rel_2).item())
    
    
def mean_confidence_interval(data_in, confidence=0.95):
    data= np.array(data_in)
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h


def evaluate_inverse_precision(ground_truth, inv_cholesky, inv_woodbury, precicion=1e-14):
    #print('compute time(percent):', (time_cholesky / time_std), flush=True)
    chol_precision = torch.all((ground_truth - inv_cholesky)**2 < precicion).item()
    wood_precision = torch.all((ground_truth - inv_woodbury)**2 < precicion).item()
    #print('truth->chol:', chol_precision, flush=True)
    #print('truth->wood:', wood_precision, flush=True)
    return chol_precision, wood_precision
    #print( torch.allclose(ground_truth, inv_cholesky, rtol=precicion) )
    #print( torch.allclose(ground_truth, inv_woodbury, rtol=precicion) )
    

def cholupdate(R,x,sign): # expects upper triangular matrix
    p = np.size(x)
    x = x.T
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k,k]**2 + x[k]**2)
        elif sign == '-':
            r = np.sqrt(R[k,k]**2 - x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        if sign == '+':
            R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        elif sign == '-':
            R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R
  
  
def cholupdate_torch(R, x, sign):
    p = x.size(0)
    x = x.view(-1)
    for k in range(p):
        if sign == '+':
            r = torch.sqrt(R[k, k]**2 + x[k]**2)
        elif sign == '-':
            r = torch.sqrt(R[k, k]**2 - x[k]**2)
        c = r / R[k, k]
        s = x[k] / R[k, k]
        R[k, k] = r
        if sign == '+':
            R[k, k+1:p] = (R[k, k+1:p] + s * x[k+1:p]) / c
        elif sign == '-':
            R[k, k+1:p] = (R[k, k+1:p] - s * x[k+1:p]) / c
        x[k+1:p] = c * x[k+1:p] - s * R[k, k+1:p]
    return R

def update_rank_one(L, x, x_copy=True):
    """
    Rank-one update: compute Chol(M + x xᵀ) given L = Chol(M).

    Remarks:
     - Update is performed in-place (so `L` is mutated).
     - We don't need to know `M` to perform the update.
    """
    start_time_init = time.time()
    if x_copy:
        x = x.clone()
    #L_nump = L.data.cpu().numpy()
    
    L = L.T
    n = x.shape[0]
    assert L.shape == (n, n)
    stop_time_init = time.time() - start_time_init
    print(stop_time_init)
    start_time_for = time.time()
    for k in range(n):
        r = torch.sqrt(L[k, k]**2 + x[k]**2)# torch.hypot(L[k, k], x[k])  # Equivalent to torch.sqrt(L[k, k]**2 + x[k]**2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k+1:, k] = (L[k+1:n, k] + s * x[k+1:n]) / c
            x[k+1:] = c * x[k+1:n] - s * L[k+1:n, k]
    stop_time_for = time.time() - start_time_for
    print(stop_time_for)
    return L.T

def cholupdate_seeger(L, x, device):
    # conversions
    #profile_time_1 = time.time()
    #L_np = L.data.cpu().numpy()
    #x_np = x.data.cpu().numpy()
    if device.type == 'cuda':
        L_np = np.asarray(L.to("cpu", non_blocking=True))
        x_np = np.asarray(x.to("cpu", non_blocking=True))
        torch.cuda.synchronize()
    else:
        L_np = L.numpy()
        x_np = x.numpy()
        #torch.cpu.synchronize()
        #profile_time_diff = time.time() - profile_time_1
        #L_np = cp.from_dlpack(L)
        #x_np = cp.from_dlpack(x)
        #profile_time_2 = time.time()
    cholupdates.rank_1.update(L_np, x_np, check_diag=False, overwrite_L=True, overwrite_v=True,
        method="seeger", impl="cython")
    torch.cuda.synchronize()
    
    if device.type == 'cuda':
        ret = torch.from_numpy(L_np).to(device, non_blocking=True)
        torch.cuda.synchronize()
    else:
        #ret = torch.from_numpy(L_np)
        ret = L
        #torch.cpu.synchronize()
    #profile_time_diff_2 = time.time() - profile_time_2
    #res_tens = torch.from_numpy(L_np).cuda()
    #profile_time_3 = time.time()
    #profile_time_diff_3 = time.time() - profile_time_3
    #print ("copy to np:" + str(profile_time_diff))
    #print ("seeger:" + str(profile_time_diff_2))
    #print ("to tensor:" + str(profile_time_diff_3))
    return ret
    #return torch.from_dlpack(L_np)
  
  
if __name__ == '__main__':
      main()