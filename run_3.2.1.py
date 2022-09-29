# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import argparse
import os
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import paddle.static as static

import visual_data
from basic_model_pdpd import DeepModel_single

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pi = np.pi


def get_args():
    parser = argparse.ArgumentParser('PINNs for 1D Poisson model', add_help=False)
    parser.add_argument('-f', type=str, default="external")
    parser.add_argument('--net_type', default='gpinn', type=str)
    parser.add_argument('--epochs_adam', default=30000, type=int)
    parser.add_argument('--save_freq', default=5000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=1000, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=False, type=bool, help="use gpu")
    parser.add_argument('--work_name', default='Poisson-1D', type=str, help="save_path")

    parser.add_argument('--Nx_EQs', default=100, type=int)
    parser.add_argument('--Nx_Val', default=200, type=int)
    parser.add_argument('--g_weight', default=0.001, type=float)
    return parser.parse_args()


class Net_single(DeepModel_single):
    def __init__(self, planes, active):
        super(Net_single, self).__init__(planes, active=active, data_norm=[0, 0])

    def out_transform(self, inn_var, out_var):
        return inn_var * (pi-inn_var) * out_var + inn_var


    def equation(self, inn_var):
        out_var = self.forward(inn_var)
        out_var = self.out_transform(inn_var, out_var)
        dudx = paddle.incubate.autograd.grad(out_var, inn_var)
        d2udx2 = paddle.incubate.autograd.grad(dudx, inn_var)
        f = 8 * paddle.sin(8 * inn_var)
        for i in range(1, 5):
            f += i * paddle.sin(i*inn_var)
        eqs = f + d2udx2
        if 'gpinn' in opts.net_type:
            # d3udx3 = paddle.incubate.autograd.grad(d2udx2, inn_var)
            # dfdx = paddle.cos(inn_var) + 4 * paddle.cos(2 * inn_var) + 9 * paddle.cos(3 * inn_var)
            # + 16 * paddle.cos(4 * inn_var) + 64 * paddle.cos(8 * inn_var)
            # g_eqs = d3udx3 + dfdx
            g_eqs = paddle.incubate.autograd.grad(eqs, inn_var)
        else:
            g_eqs = paddle.zeros((1,), dtype=paddle.float32)
        return eqs, g_eqs


def build(opts, model):
    ## 采样

    # print(out_BCs)

    EQs_var = paddle.static.data('EQs_var', shape=[opts.Nx_EQs, 1], dtype='float32')
    EQs_var.stop_gradient = False
    # EQs_tar = paddle.static.data('EQs_tar', shape=[opts.Nx_EQs, 1], dtype='float32')


    Val_var = paddle.static.data('Val_var', shape=[opts.Nx_Val, 1], dtype='float32')
    Val_var.stop_gradient = False
    Val_tar = paddle.static.data('Val_tar', shape=[opts.Nx_Val, 1], dtype='float32')

    eqs, g_eqs = model.equation(EQs_var)
    val = model(Val_var)
    val = model.out_transform(Val_var, val)
    val_grad = paddle.incubate.autograd.grad(val, Val_var)
    val_equa, _ = model.equation(Val_var)

    # print(fields_all)

    EQsLoss = paddle.norm(eqs, p=2) ** 2 / opts.Nx_EQs  # 方程所有计算守恒残差点的损失，不参与训练
    gEQsLoss = paddle.norm(g_eqs, p=2) ** 2 / opts.Nx_EQs  # 方程所有计算守恒残差点的损失，不参与训练
    datLoss = paddle.norm(Val_tar - val, p=2) ** 2 / opts.Nx_Val  # 方程所有计算守恒残差点的损失，不参与训练

    total_loss = EQsLoss + gEQsLoss * opts.g_weight

    scheduler = paddle.optimizer.lr.MultiStepDecay(0.001, [opts.epochs_adam*0.6, opts.epochs_adam*0.8], gamma=0.1)
    optimizer = paddle.optimizer.Adam(scheduler)
    optimizer.minimize(total_loss)
    # optimizer.minimize(EQsLoss)
    #
    return [val, val_grad, val_equa], [EQsLoss, gEQsLoss, datLoss, total_loss], scheduler


# 解析解 ve=1e-3
def sol(x):
    solution = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        solution += 1 / i * np.sin(i * x)
    return solution


def grad(x):
    solution = 1 + np.cos(8 * x)
    for i in range(1, 5):
        solution += np.cos(i * x)
    return solution


def gen_all(num):
    xvals = np.linspace(0, pi, num, dtype=np.float32)
    yvals = sol(xvals)
    ygrad = grad(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1)), np.reshape(ygrad, (-1, 1))


if __name__ == '__main__':

    opts = get_args()

    try:
        import paddle.fluid as fluid
        use_cuda = opts.device
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    except:
        place = None

    paddle.enable_static()

    # opts.Nx_EQs = N
    save_path = opts.net_type + '-Nx_EQs_' + str(opts.Nx_EQs)
    work_path = os.path.join('work', opts.work_name, save_path)
    tran_path = os.path.join('work', opts.work_name, save_path, 'train')
    isCreated = os.path.exists(tran_path)
    if not isCreated:
        os.makedirs(tran_path)

    # 将控制台的结果输出到a.log文件，可以改成a.txt
    sys.stdout = visual_data.Logger(os.path.join(work_path, 'train.log'), sys.stdout)
    print(opts)


    train_x, train_u, _ = gen_all(opts.Nx_EQs)
    valid_x, valid_u, valid_g = gen_all(opts.Nx_Val)


    paddle.incubate.autograd.enable_prim()

    planes = [1, ] + [20, ] * 3 + [1, ]
    Net_model = Net_single(planes=planes, active=nn.Tanh())
    [U_pred, U_grad, U_eqs], Loss, Scheduler = build(opts, Net_model)

    exe = static.Executor(place)
    exe.run(static.default_startup_program())
    prog = static.default_main_program()

    Visual = visual_data.matplotlib_vision('/', field_name=('u',), input_name=('x',))
    star_time = time.time()
    log_loss = []
    start_epoch = 0

    for epoch in range(start_epoch, 1+opts.epochs_adam):
        ## 采样
        Scheduler.step()
        exe.run(prog, feed={'EQs_var': train_x, 'Val_var': valid_x, 'Val_tar': valid_u}, fetch_list=[Loss[-1]])

        if epoch > 0 and epoch % opts.print_freq == 0:
            all_items = exe.run(prog, feed={'EQs_var': train_x, 'Val_var': valid_x, 'Val_tar': valid_u},
                                fetch_list=[[U_pred, U_grad, U_eqs] + Loss[:-1]])

            u_pred = all_items[0]
            u_grad = all_items[1]
            u_eqs = all_items[2]
            loss_items = all_items[3:]

            log_loss.append(np.array(loss_items).squeeze())

            print('iter: {:6d}, lr: {:.1e}, cost: {:.2f}, val_loss: {:.2e}, Eqs_loss: {:.2e}, Grad_loss: {:.2e}'.
                  format(epoch, Scheduler.get_lr(), time.time() - star_time, float(loss_items[-1]),
                         float(loss_items[0]), float(loss_items[1])))
            star_time = time.time()
            # print(np.array(par_pred).shape)
        if epoch > 0 and epoch % opts.save_freq == 0:
            plt.figure(1, figsize=(20, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            if "gpinn" in opts.net_type:
                Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'grad_loss')
            plt.savefig(os.path.join(tran_path, 'log_loss.svg'))

            plt.figure(4, figsize=(20, 15))
            plt.clf()
            Visual.plot_value(valid_x, valid_u, 'EXACT')
            Visual.plot_value(valid_x, u_pred, opts.net_type)
            plt.xlabel("x")
            plt.ylabel("u")
            plt.savefig(os.path.join(tran_path, 'pred_u.svg'))

            plt.figure(5, figsize=(20, 15))
            plt.clf()
            Visual.plot_value(valid_x, valid_g, 'EXACT')
            Visual.plot_value(valid_x, u_grad, opts.net_type)
            plt.xlabel("x")
            plt.ylabel("du/dx")
            plt.savefig(os.path.join(tran_path, 'grad_u.svg'))


            star_time = time.time()

            paddle.save({'log_loss': log_loss,
                         'valid_x': valid_x, 'valid_u': valid_u, 'valid_g': valid_g,
                         'u_pred': u_pred, 'u_grad': u_grad, 'u_eqs': u_eqs,
                         }, os.path.join(work_path, 'out_res.pth'), )
    paddle.save(prog.state_dict(), os.path.join(work_path, 'latest_model.pdparams'), )
    time.sleep(3)
    shutil.move(os.path.join(work_path, 'train.log'), tran_path)
