
import argparse
import os
import shutil
import sys
import time
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import paddle.static as static

import visual_data
from basic_model_pdpd import DeepModel_single
from SALib.sample import sobol_sequence

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pi = np.pi
def get_args():
    parser = argparse.ArgumentParser('PINNs for Brinkman-Forchheimer model', add_help=False)
    parser.add_argument('-f', type=str, default="external")
    parser.add_argument('--net_type', default='gpinn', type=str)
    parser.add_argument('--epochs_adam', default=60000, type=int)
    parser.add_argument('--save_freq', default=2000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=200, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=True, type=bool, help="use gpu")
    parser.add_argument('--work_name', default='Brinkman-Forchheimer-1', type=str, help="save_path")

    parser.add_argument('--Nx_EQs', default=30, type=int)
    parser.add_argument('--Nx_Sup', default=5, type=int)
    parser.add_argument('--Nx_Val', default=500, type=int)
    parser.add_argument('--g_weight', default=0.1, type=float)
    return parser.parse_args()

def sol(x):
    r = (v * e / (1e-3 * K)) ** (0.5)
    return g * K / v * (1 - np.cosh(r * (x - H / 2)) / np.cosh(r * H / 2))

def gen_sup(num):
    xvals = np.linspace(1 / (num + 1), 1, num, dtype=np.float32, endpoint=False)
    yvals = sol(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1))



opts = get_args()

g = 1
v = 1e-3
K = 1e-3
e = 0.4
H = 1
sup_x, sup_u = gen_sup(opts.Nx_Sup)

log_err = []
log_res = []
log_par = []
Ns = [5, 10, 15, 20, 25, 30]
Nets = ['pinn'] #['pinn', 'gpinn']
for N in Ns:
    for net_type in Nets:
        u_err = []
        g_err = []
        par_err = []
        for id in range(1, 10):
            opts.Nx_EQs = N
            opts.net_type = net_type
            save_path = opts.net_type + '+' + str(id) + '-Nx_EQs_' + str(opts.Nx_EQs)
            work_path = os.path.join('work', opts.work_name, save_path)
            vald_path = os.path.join('work', opts.work_name, 'valid')
            isCreated = os.path.exists(vald_path)
            if not isCreated:
                os.makedirs(vald_path)
            # 将控制台的结果输出到a.log文件，可以改成a.txt
            Visual = visual_data.matplotlib_vision('/', field_name=('u',), input_name=('x',))
            sys.stdout = visual_data.Logger(os.path.join(vald_path, 'valid.log'), sys.stdout)
            cp = paddle.load(os.path.join(work_path, 'out_res.pth'))
            log_loss = np.array(cp['log_loss'])
            par_pred = np.array(cp['par_pred'])
            valid_x = np.array(cp['valid_x'])
            valid_u = np.array(cp['valid_u'])
            valid_g = np.array(cp['valid_g'])
            u_pred = np.array(cp['u_pred'])
            u_grad = np.array(cp['u_grad'])

            par_err.append(np.abs(par_pred[-1] - 1e-3)/1e-3)
            u_err.append(np.linalg.norm(u_pred - valid_u)/np.linalg.norm(valid_u))
            g_err.append(np.linalg.norm(u_grad - valid_g)/np.linalg.norm(u_grad))


            if N == 10 and id == 1:
                log_res.append([valid_x, valid_u, u_pred])
            if N == 15 and id == 1:
                log_par.append([par_pred])

        log_err.append([np.mean(np.array(par_err)), np.std(np.array(par_err)) / 2,
                        np.mean(np.array(u_err)), np.std(np.array(u_err)) / 2,
                        np.mean(np.array(g_err)), np.std(np.array(g_err)) / 2, ])

log_err = np.reshape(np.array(log_err), [len(Ns), len(Nets), -1])
ylabels = ['L2 relative error of v_e', 'L2 relative error of u', 'L2 relative error of du/dx']
fignams = ['/Fig6_A.jpg', '/Fig6_B.jpg', '/Fig6_C.jpg']
for i in range(len(ylabels)):
    plt.figure(i+1, figsize=(10, 8))
    plt.clf()
    plt.semilogy(np.array(Ns), log_err[:, 0, i*2], 'b-o', label='PINN')
    plt.fill_between(np.array(Ns), log_err[:, 0, i*2] - log_err[:, 0, i*2+1], log_err[:, 0, i*2] + log_err[:, 0, i*2+1], color='b', alpha=0.3)
    plt.semilogy(np.array(Ns), log_err[:, 1, i*2], 'r-o', label='GPINN')
    plt.fill_between(np.array(Ns), log_err[:, 1, i*2] - log_err[:, 1, i*2+1], log_err[:, 1, i*2] + log_err[:, 1, i*2+1], color='r', alpha=0.3)
    plt.xlabel('No. of training points')
    plt.ylabel(ylabels[i])
    plt.legend()
    plt.xlim([5, 35])
    plt.rcParams['font.size'] = 20
    plt.savefig(vald_path + fignams[i])


plt.figure(4, figsize=(8, 6))
plt.clf()
plt.plot(valid_x, log_res[-2][0], '-k', linewidth=2, label='EXACT')
plt.plot(valid_x, log_res[-2][1], '--b', linewidth=2, label='PINN')
plt.plot(valid_x, log_res[-1][1], '--r', linewidth=2, label='GPINN')
plt.plot(sup_x, sup_u, c='k', marker='s', label='Observed')
plt.xlabel('x')
plt.ylabel('u')
plt.rcParams['font.size'] = 20
plt.legend(frameon=False)
plt.savefig(vald_path + '/Fig6_D-N' + str(opts.Nx_EQs) + '.jpg')

plt.figure(5, figsize=(8, 6))
plt.clf()
Visual.plot_loss(np.arange(len(log_par[-1])), np.ones_like(log_par[-1]) * 1e-3, label='EXACT')
Visual.plot_loss(np.arange(len(log_par[-1])), np.array(log_par)[-2], label='PINN')
Visual.plot_loss(np.arange(len(log_par[-1])), np.array(log_par)[-1], label='GPINN')
plt.xlabel('Epoch')
plt.ylabel('v_e')
plt.rcParams['font.size'] = 20
plt.savefig(vald_path + '/Fig6_E-N' + str(opts.Nx_EQs) + '.jpg')








