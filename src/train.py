import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
import os
import pynvml
import utils

def unnormalize(x, MAX, MIN):
    return (x * (MAX-MIN)/2 + (MAX+MIN)/2).clamp(MIN, MAX)


def SMAPE(pred, target):
    t = target[target>2]
    p = pred[target>2]
    return torch.sum(2.0 * (t - p).abs() / (t.abs() + p.abs())).item(), target[target!=0].numel()


class TrainLoop:
    def __init__(self, args, writer, model, data, train_index, test_index, val_index, device, best_rmse=1e9):
        self.args = args
        self.writer = writer
        self.model = model
        self.data = data
        self.train_data = train_index
        self.test_data = test_index
        self.val_data = val_index
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.best_rmse_random = 1e9
        self.warmup_steps=1
        self.min_lr = args.min_lr
        self.best_rmse = best_rmse
        self.early_stop = 0

    def run_step(self, batch, step, mask_ratio, mask_strategy,index, name, topo = None, subgraphs = None):
        self.opt.zero_grad()
        loss, num, loss_real, num2, loss_real_nonmask = self.forward_backward(batch, step, mask_ratio, mask_strategy,index=index, name = name, topo = topo, subgraphs = subgraphs)

        self._anneal_lr()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
        self.opt.step()
        return loss, num, loss_real, num2, loss_real_nonmask

    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0, Type='val'):
        
        with torch.no_grad():
            error_smape, error_mae, error_norm, error, num, error2, num2, num_smape = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            error_test, error2_test, error_mae_test = 0.0, 0.0, 0.0
            matrix = test_data[index][1]
            subgraphs = test_data[index][2]

            data, ts = self.data[dataset]
            
            for iindex, batch in enumerate(test_data[index][0]):

                assert batch.shape[1]==3

                value = torch.stack([data[i:i+self.args.seq_len] for i in batch[:,0]]).unsqueeze(dim=1)

                timestamps = torch.stack([ts[i:i+self.args.seq_len] for i in batch[:,0]])
                batch = (value, timestamps)

                loss, loss2, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data = dataset, mode='forward',topo = matrix, subgraphs = subgraphs)

                pred = torch.clamp(pred, min=-1, max=1)

                pred_mask = pred.squeeze(dim=2)
                target_mask = target.squeeze(dim=2)

                if self.args.data_norm==1:
                    
                    error += mean_squared_error(self.args.scaler[dataset].inverse_transform(pred_mask[mask==1].reshape(-1,1).detach().cpu().numpy()), self.args.scaler[dataset].inverse_transform(target_mask[mask==1].reshape(-1,1).detach().cpu().numpy()), squared=True) * mask.sum().item()
                    error2 +=  mean_squared_error(self.args.scaler[dataset].inverse_transform(pred_mask[mask==0].reshape(-1,1).detach().cpu().numpy()), self.args.scaler[dataset].inverse_transform(target_mask[mask==0].reshape(-1,1).detach().cpu().numpy()), squared=True) * (1-mask).sum().item()
                    error_mae += mean_absolute_error(self.args.scaler[dataset].inverse_transform(pred_mask[mask==1].reshape(-1,1).detach().cpu().numpy()), self.args.scaler[dataset].inverse_transform(target_mask[mask==1].reshape(-1,1).detach().cpu().numpy())) * mask.sum().item()


                    error_norm += loss.item() * mask.sum().item()

                num += mask.sum().item()
                num2 += (1-mask).sum().item()

        rmse = np.sqrt(error / num)
        mae = error_mae / num
        loss_test = error_norm / num

        return rmse, mae, loss_test


    def Evaluation(self, test_data, epoch, seed=None, best=True, Type='val'):
        
        loss_list = []

        rmse_list = []
        rmse_key_result = {}

        self.model.eval()

        for index, dataset_name in enumerate(self.args.dataset.split('*')):

            rmse_key_result[dataset_name] = {}

            s = self.args.mask_strategy
            m = self.args.mask_ratio
            result, mae,  loss_test = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index, Type=Type)
            rmse_list.append(result)
            loss_list.append(loss_test)
            if s not in rmse_key_result[dataset_name]:
                rmse_key_result[dataset_name][s] = {}
            rmse_key_result[dataset_name][s][m] = {'rmse':result, 'mae':mae, 'smape':smape}
            
            if Type == 'val':
                self.writer.add_scalar('Evaluation/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), result, epoch)
            elif Type == 'test':
                self.writer.add_scalar('Test_RMSE/{}'.format(sdataset_name), result, epoch)
                self.writer.add_scalar('Test_MAE/{}'.format(dataset_name), mae, epoch)

        loss_test = np.mean(loss_list)

        self.model.train()

        if best:
            return self.best_model_save(epoch, loss_test, rmse_key_result)

        return loss_test, rmse_key_result

    def best_model_save(self, step, rmse, rmse_key_result):
        if rmse < self.best_rmse:
            self.early_stop = 0
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            self.best_rmse = rmse
            self.writer.add_scalar('Evaluation/RMSE_best', self.best_rmse, step)
            print('\nRMSE_best:{}\n'.format(self.best_rmse))
            print(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('epoch:{}, best rmse: {}\n'.format(step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('epoch:{}, best rmse: {}\n'.format(step, self.best_rmse))
                f.write(str(rmse_key_result)+'\n')
            return 'save'

        else:
            self.early_stop += 1
            print('\nRMSE:{}, RMSE_best:{}, early_stop:{}\n'.format(rmse, self.best_rmse, self.early_stop))
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('RMSE:{}, not optimized, early_stop:{}\n'.format(rmse, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                with open(self.args.model_path+'result.txt', 'a') as f:
                    f.write('Early stop!\n')
                with open(self.args.model_path+'result_all.txt', 'a') as f:
                    f.write('Early stop!\n')
                    exit()

            return 'none'
        
    def mask_select(self):
        mask_strategy = self.args.mask_strategy
        mask_ratio = self.args.mask_ratio
        return mask_strategy, mask_ratio


    def run_loop(self):
        step = 0
        if self.args.mode=='testing' or self.args.mode=='zero':
            #self.Evaluation(self.val_data, 0, best=True, Type='val')
            self.Evaluation(self.test_data, 0, best=False, Type='test')
            exit()
        for epoch in range(self.args.total_epoches):
            print('Training')
            self.step = epoch
            
            loss_all, num_all, loss_real_all, num_all2, loss_real_nonmask_all = 0.0, 0.0,0.0, 0.0, 0.0
            start = time.time()

            for name, batch_index, matrix, subgraphs in self.train_data:
                mask_strategy, mask_ratio = self.mask_select()
                batch, ts = self.data[name]

                assert batch_index.shape[1]==3
                
                value = torch.stack([batch[i:i+self.args.seq_len] for i in batch_index[:,0]]).unsqueeze(dim=1)
                timestamps = torch.stack([ts[i:i+self.args.seq_len] for i in batch_index[:,0]])
                batch = (value, timestamps)
                loss, num, loss_real, num2, loss_real_nonmask  = self.run_step(batch, step, mask_ratio=mask_ratio, mask_strategy = mask_strategy,index=0, name = name, topo=matrix, subgraphs = subgraphs)
                step += 1
                loss_all += loss * num
                loss_real_all += loss_real * num
                num_all += num
                num_all2 += num2
                loss_real_nonmask_all += loss_real_nonmask * num2
            
            end = time.time()
            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}, training rmse:{}, training rmse nonmask:{}'.format(epoch, loss_all / num_all,np.sqrt(loss_real_all / num_all), np.sqrt(loss_real_nonmask_all / num_all2)))

            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('epoch:{}, training loss:{}, training rmse:{}, training rmse nonmask:{}\n'.format(epoch, loss_all / num_all,np.sqrt(loss_real_all / num_all), np.sqrt(loss_real_nonmask_all / num_all2)))

            if epoch >= 10 or self.args.mode!='training':
                self.writer.add_scalar('Training/Loss_epoch', loss_all / num_all, epoch)
                self.writer.add_scalar('Training/Loss_real', np.sqrt(loss_real_all / num_all), epoch)
                self.writer.add_scalar('Training/Loss_real_nonmask', np.sqrt(loss_real_nonmask_all / num_all2), epoch)

            if (epoch+1) % self.log_interval == 0 or epoch == self.args.total_epoches-1:
                print('Evaluation')
                is_break = self.Evaluation(self.val_data, epoch, best=True, Type='val')

                if is_break == 'save':
                    print('test evaluate!')
                    rmse_test, rmse_key_test = self.Evaluation(self.test_data, epoch, best=False, Type='test')
                    print('epoch:{}, test rmse: {}\n'.format(epoch, rmse_test))
                    print(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result.txt', 'a') as f:
                        f.write('\n\nepoch:{}, test rmse: {}\n\n'.format(epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')
                    with open(self.args.model_path+'result_all.txt', 'a') as f:
                        f.write('\n\nepoch:{}, test rmse: {}\n\n'.format(epoch, rmse_test))
                        f.write(str(rmse_key_test)+'\n')

        return self.best_rmse

    def model_forward(self, batch, model, mask_ratio, mask_strategy, seed=None, data=None, mode='backward',topo = None, subgraphs = None):

        batch = [i.to(self.device) for i in batch]

        if topo is not None:
            topo = topo.to(self.device)

        patch_size = utils.select_patch_size(self.args, data)

        loss, loss2, pred, target, mask = self.model(
                batch,
                mask_ratio=mask_ratio,
                mask_strategy = mask_strategy, 
                seed = seed, 
                data = data,
                mode = mode,
                topo = topo,
                subgraphs = subgraphs,
                patch_size = patch_size
            )
        return loss, loss2, pred, target, mask 

    def forward_backward(self, batch, step, mask_ratio, mask_strategy,index, name=None, topo=None, subgraphs = None):

        loss, loss2, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, data=name, mode='backward',topo = topo, subgraphs = subgraphs)

        if self.args.data_norm == 1:
            pred_mask = pred.squeeze(dim=2)[mask==1]
            target_mask = target.squeeze(dim=2)[mask==1]

            loss_real = torch.mean((self.args.scaler[name].inverse_transform(pred_mask.reshape(-1,1)) - self.args.scaler[name].inverse_transform(target_mask.reshape(-1,1)))**2).item()
            pred_nonmask = pred.squeeze(dim=2)[mask==0]
            target_nonmask = target.squeeze(dim=2)[mask==0]
            loss_real_nonmask = torch.mean((self.args.scaler[name].inverse_transform(pred_nonmask.reshape(-1,1)) - self.args.scaler[name].inverse_transform(target_nonmask.reshape(-1,1)))**2).item()

        else:
            loss_real = loss.item()
            loss_real_nonmask = loss2.item()

        loss.backward()

        self.writer.add_scalar('Training/Loss_step', np.sqrt(loss_real), step)
        return loss.item(), mask.sum().item(), loss_real, (1-mask).sum().item(), loss_real_nonmask

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
        elif self.step < self.lr_anneal_steps:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.step - self.warmup_steps)
                    / (self.lr_anneal_steps - self.warmup_steps)
                )
            )
        else:
            lr = self.min_lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.writer.add_scalar('Training/LR', lr, self.step)
        return lr


def print_unused_layers(model, x):
    layers_called = set()

    def register_hook(module):
        def hook(module, input, output):
            layers_called.add(module)
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only register hook on leaf modules
            module.register_forward_hook(register_hook(module))

    with torch.no_grad():
        model(x)

    module_list = []

    for name, module in model.named_modules():
        if module not in layers_called:
            module_list.append(name.split('.')[0])
            print(f"Layer {name} was not called in forward pass")

    print(set(module_list))