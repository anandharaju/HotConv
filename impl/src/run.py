import numpy as np
import pandas as pd
from model.model import get_malconv_variant
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import tqdm
from utilz.gpu_usage import GPU_Usage
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, top_k_accuracy_score
import gc
import warnings
warnings.filterwarnings("ignore")
from actions.validate import validate
from utilz.utilz import Util
from utilz.carbonator import carbon_footprint
import psutil
import os
from torchsummary import summary


# Original MalConv
def train_original(malconv, exe_input, label, args, ghandle):
    # Forward for top-activation data
    outputs = malconv(exe_input.long().to(args.device))
    loss = args.loss_fn(outputs, label)
    loss.backward()
    args.adam_optim.step()
            
    _, predicted = torch.max(outputs.data, 1)
    with torch.no_grad():
        args.preds.extend(torch.softmax(outputs, dim=-1).data.detach().cpu().numpy())
        args.truths.extend(label.detach().cpu().numpy().ravel())

    args.train_correct += (predicted == label).sum().item()
    args.train_total += label.size(0)
    args.cur_batch_loss = loss.item()
    args.train_loss += outputs.shape[0] * args.cur_batch_loss  # per batch
    return args


# Adaptation of MalConv with Chunking Strategy
def train_malconv2(malconv, exe_input, label, args, ghandle):
    xt = time.time()
    outputs, args = malconv(exe_input, ghandle, args)
    args.time_fp_all += time.time() - xt

    losst = time.time()
    loss = args.loss_fn(outputs, label)
    args.time_loss += time.time() - losst
    # print("Loss time:", time.time()-fp2t)
    
    xt = time.time()
    loss.backward()
    # print("M2 BP  time:", time.time()-xt)
    args.time_bp += time.time() - xt
    
    xt = time.time()
    args.adam_optim.step()
    # print("M2 Opt time:", time.time()-xt)
    args.time_optim += time.time() - xt
    
    ot = time.time()
    _, predicted = torch.max(outputs.data, 1)
    with torch.no_grad():
        args.preds.extend(torch.softmax(outputs, dim=-1).data.detach().cpu().numpy())
        args.truths.extend(label.detach().cpu().numpy().ravel())
    args.train_correct += (predicted == label).sum().item()
    args.train_total += label.size(0)
    args.cur_batch_loss = loss.item()
    args.train_loss += outputs.shape[0] * args.cur_batch_loss  # per batch 
    args.time_others += time.time() - ot
    return args


def train_proposed(malconv, exe_input, label, args, ghandle):
    # Forward for top-activation data
    fpt = time.time()
    malconv.fp_slice_size = args.fp_slice_size
    outputs, fcache = malconv(exe_input, ghandle)
    args.time_fp += time.time() - fpt
    # print("FP time:", time.time() - fpt)

    xt = time.time()
    batch_blocks = malconv.gather_hot_blocks(fcache, exe_input)
    #print("Hot block time:", time.time() - xt)
    args.time_hot += time.time() - xt
    #print("FP + Hot block time:", time.time()-fpt)
    args.time_fp_hot += time.time() - fpt
    
    '''
    # FOR FORWARD PASS - 2
    # IF CONVOLUTION WINDOW STRIDE IS NOT SAME AS THE CONVOLUTION WINDOW SIZE,
    # TEMPORARILY SET THE STRIDE SAME AS THE CONVOLUTION WINDOW SIZE, RESET AFTER FORWARD PASS - 2
    malconv.conv_1.stride = args.window_size
    malconv.conv_2.stride = args.window_size
    fp2t = time.time()
    outputs, fcache = malconv.forward_for_cache(batch_blocks, ghandle)
    args.time_fp2 += time.time() - fp2t
    #print("Returned to call: ", time.time() - xt)
    malconv.conv_1.stride = args.stride # RESTORE THE STRIDE NEEDED FOR FORWARD PASS - 1
    malconv.conv_2.stride = args.stride
    '''
    
    losst = time.time()
    loss = args.loss_fn(outputs, label)
    args.time_loss += time.time() - losst
    #print("FP2 + loss calc time:", time.time()-fp2t)
    
    ot = time.time()
    _, predicted = torch.max(outputs.data, 1)
    with torch.no_grad():
        args.preds.extend(torch.softmax(outputs, dim=-1).data.detach().cpu().numpy())
        args.truths.extend(label.detach().cpu().numpy().ravel())

    args.train_correct += (predicted == label).sum().item()
    args.train_total += label.size(0)
    args.cur_batch_loss = loss.item()
    args.train_loss += outputs.shape[0] * args.cur_batch_loss  # per batch
    malconv.var = None
    args.time_others += time.time() - ot

    xt = time.time()
    loss.backward()
    args.time_dense_bp += time.time()-xt
    #print("Dense BP time:", time.time()-xt)
    
    xt = time.time()
    malconv.backprop_selective_gradient_attribution(args.adam_optim, fcache, batch_blocks, ghandle)
    args.time_conv_bp += time.time()-xt
    #print("Dense BP + Conv BP + OPT  time:", time.time()-xt)
    
    # args.time_bp += time.time() - xt
    del fcache
            
    return args
            
def train(malconv, exe_input, label, args, ghandle):
    trainer = None
    if args.variant == 'proposed':
        trainer = train_proposed
    elif args.variant == 'malconv2':
        trainer = train_malconv2
    elif args.variant == 'malconv':
        trainer = train_original
    return trainer(malconv, exe_input, label, args, ghandle)

def run(args, fold, util, ghandle, cvres):
    print("\n\n[[[  Fold", str(fold), "Begins  ]]]")
    args = util.prepare_data(args, fold)
    malconv = get_malconv_variant(args)
    args.adam_optim = optim.AdamW([{'params': malconv.parameters()}]) # , lr=args.learning_rate)
    args.loss_fn = nn.CrossEntropyLoss(weight=args.class_weights, reduction='mean').to(args.device)
    ghandle.get_gpu_usage("AFTER MODEL INSTANTIATION")

    epoch = 0
    cur_topkacc = 0
    criteria = np.inf
    train_time = 0
    # ghandle.get_gpu_usage("BEFORE START OF TRAINING")
    cur_patience = args.early_stopping_patience
    drill_time = {'data_load': 0, 'fp': 0, 'fp2': 0, 'hotblocks': 0, 'bp': 0}

    while not args.test_only_mode and epoch < args.epochs:
        args.time_fp = 0
        args.time_hot = 0
        args.time_fp_hot = 0
        args.time_fp2 = 0
        args.time_fp_all = 0
        args.time_bp = 0
        args.time_dense_bp = 0
        args.time_conv_bp = 0
        args.time_loss = 0
        args.time_optim = 0
        args.time_load = 0
        args.time_others = 0
        args.time_total = 0

        args.preds = []
        args.truths = []
        epoch_stats = {'epoch': epoch}
        train_epoch_start_time = time.time()
        args.train_correct = 0
        args.train_total = 0
        args.train_loss = 0.0
        itr = 0
        tbatch_times = []
        vbatch_times = []
        
        lt = time.time()
        tlt = time.time()

        '''if epoch % 2 == 0:
            # Load smaller files
        else:
            # Load larger files
            args.dataloader.dataset.corpus_ram
            args.seqdataset_train.all_files''' 
                                               
        for batch_data in tqdm.tqdm(args.dataloader):
            if args.subloader_batch_size >= args.batch_size:
                exe_input = batch_data[0]
                label = batch_data[1].to(args.device)
            '''else:
                if itr == 0:
                    exe_input = batch_data[0]
                    label = batch_data[1].to(args.device)
                    itr += 1
                    continue
                else:
                    exe_input, label = args.seqdataset_train.concat_var_len_batches(exe_input, label, batch_data, args.device, args.batch_padding)
                    itr += 1
                    if itr == args.batch_size / args.subloader_batch_size:
                        itr = 0
                    else:
                        continue'''
            
            '''print("1", exe_input.shape, len(exe_input[0]) % args.window_size)
            exe_input = exe_input.long()
            if len(exe_input[0]) % args.window_size != 0:
              residue = args.window_size - (len(exe_input[0]) % args.window_size)
              exe_input = F.pad(exe_input, (0, residue), value=0)
            print("2", exe_input.shape, len(exe_input[0]) % args.window_size)'''

            #print("1", exe_input.shape, len(exe_input[0]) % args.fp_slice_size)
            exe_input = exe_input.long()
            if len(exe_input[0]) % args.fp_slice_size != 0:
                residue = args.fp_slice_size - (len(exe_input[0]) % args.fp_slice_size)
                exe_input = F.pad(exe_input, (0, residue), value=0)
            #print("2", exe_input.shape, len(exe_input[0]) % args.fp_slice_size)

            tbt = time.time()
            tmx = exe_input.shape[1]
            #print(exe_input.shape[1])
            label = Variable(label[:, 0].long(), requires_grad=False)  # ***********************       bodmas  - 1
            args.adam_optim.zero_grad()
            
            lt = time.time() - lt
            args.time_load += lt
            #if len(tbatch_times) <= 2:
            #    print("Loading Time:", lt)
            # tlt += lt

            args = train(malconv, exe_input, label, args, ghandle)
            ot = time.time()
            tbatch_times.append((time.time() - tbt, tmx, args.cur_batch_loss))
            if len(tbatch_times) <= 5:
                ghandle.get_gpu_usage(args.variant + ": AFTER Optimization Step")
            args.time_others += time.time() - ot
            # print("run", time.time() - ot)
            lt = time.time()

        args.time_total = time.time() - tlt

        epoch_stats['tr_acc'] = args.train_correct * 1.0 / args.train_total
        args.preds = torch.Tensor(args.preds)
        row_sums = torch.sum(args.preds, 1)
        args.preds = torch.div(args.preds , row_sums.unsqueeze(dim=1))
        predictions = np.argmax(args.preds, axis=1)
        
        epoch_stats['tr_auc'] = roc_auc_score(args.truths, predictions if args.num_classes==2 else args.preds, multi_class='ovr')
        epoch_stats['tr_loss'] = args.train_loss / args.train_population
        train_topkacc = top_k_accuracy_score(args.truths, predictions if args.num_classes==2 else args.preds, k=args.topk, labels=list(range(args.num_classes)), sample_weight=args.train_sample_weights)    
        gc.collect()
        train_time += time.time() - train_epoch_start_time

        args.cpu_load_avg = np.append(args.cpu_load_avg, psutil.getloadavg()[2])

        # VALIDATION ######################################################################################
        vbatch_times, epoch_stats, val_res = validate(vbatch_times, epoch_stats, args, ghandle, mode='val', model=malconv)
        if criteria >= epoch_stats['val_loss']:  # val_res['prc_auc']:  # epoch_stats['val_loss']
            criteria = epoch_stats['val_loss']
            cur_topkacc = val_res['topkacc']
            mpath = ("../model/"
                     + str(args.variant)
                     + "_fold" + str(fold)
                     + "_b" + str(args.batch_size)
                     + "_f" + str(args.num_filters)
                     + "_l" + str(args.max_len))
            torch.save(malconv.state_dict(), mpath)
            malconv.load_state_dict(torch.load(mpath))
            print("Saving best model...")
            cur_patience = args.early_stopping_patience
            cvres['res']['val'] = val_res
        else:
            cur_patience -= 1
        
        print("EPOCH " + str(epoch) + " Results", "F:", fold, "P:", cur_patience, "TLoss:", str(epoch_stats['tr_loss'])[:5], "\tVLoss", str(epoch_stats['val_loss'])[:5], "\tTTop-"+str(args.topk)+" Acc: ", str(train_topkacc)[:6], "\tVTop-"+str(args.topk)+" Acc: ", str(cur_topkacc)[:6], "\tTAUC:", str(epoch_stats['tr_auc'])[:6], "\tVAUC:", str(epoch_stats['val_auc'])[:6])
        ghandle.get_gpu_usage("VALIDATION: End")
        if cur_patience <= 0:
            print("Early stopping . . . !")
            break
            
        epoch += 1
        print()

    cvres['ValTop'+str(args.topk)+'-Acc'].append(cur_topkacc)


    # TESTING ######################################################################################
    del args.train_corpus
    del args.val_corpus
    args.seqdataset_test.corpus_ram = util.load_pickled(args.dpath+args.dataset+"/"+"test_corpus.pkl")

    tbatch_times, _, tst_res = validate(tbatch_times, None, args, ghandle, mode='test', model=malconv)
    cvres['TestTop'+str(args.topk)+'-Acc'].append(tst_res['topkacc'])
    print("Test Top-k Accuracy        :\t", tst_res['topkacc'], "\t\tTest F1-score (weighted)   :\t", tst_res['f1'])

    cvres['res']['test'] = tst_res
    return args, cvres, train_time / (epoch + 1), epoch, drill_time


def grid_run(args, ghandle):  
    
    util.display_args()
    cvres = {'ValTop' + str(args.topk) + '-Acc': [], 'TestTop' + str(args.topk) + '-Acc': [], 'res': {}}
    
    if not args.test_only_mode:
        # Cross Validation
        for fold in range(0, args.folds):
            args, cvres, avg_train_epoch_time, epoch, drill_time = run(args, fold, util, ghandle, cvres)
            args.early_stopping_patience = args.early_stopping_patience_
            print("\n\nCumulative Cross validation Results:    [Fold-" + str(fold) + "]")
            for key in cvres.keys():
                if key != 'res':  # in ['res', 'val', 'tst', 'cpu_tst']:
                    print(key, ":\t", np.sum(cvres[key]) / (fold + 1), "\t+/-", np.std(cvres[key]))


        exetime = time.time() - st
        '''
        CPUusage = args.cpu_load_avg.mean() / psutil.cpu_count()
        GPUusage = ghandle.get_gpu_usage_factor_()
        cfp = carbon_footprint()
        cfp = cfp.get_carbonEmissions(actual_runTime_hours=(exetime / (60 * 60)), usageGPU_used=GPUusage)
        '''
        
        
        for key in cvres['res'].keys():
            cvres['res'][key]['avg_train_epoch_time'] = avg_train_epoch_time
            cvres['res'][key]['total_training_time'] = exetime / 60
            cvres['res'][key]['total_training_epochs'] = epoch
            #cvres['res'][key]['carbon'] = cfp
            cvres['res'][key]['grmmpeak'] = ghandle.get_grmmpeak_()
            cvres['res'][key]['gpu_total_usage'] = ghandle.get_total_usage_()
            cvres['res'][key]['gpu_usage_factor'] = GPUusage
            cvres['res'][key]['cpu_usage_factor'] = CPUusage
            cvres['res'][key]['avg_data_load'] = drill_time['data_load'] / epoch
            cvres['res'][key]['avg_fp'] = drill_time['fp'] / epoch
            cvres['res'][key]['avg_fp2'] = drill_time['fp2'] / epoch
            cvres['res'][key]['avg_hotblocks'] = drill_time['hotblocks'] / epoch
            cvres['res'][key]['avg_bp'] = drill_time['bp'] / epoch
            cvres['res'][key]['fold'] = fold
            cvres['res'][key]['device'] = args.device
            
            pd.DataFrame.from_dict(data=[cvres['res'][key]]).to_csv("../out/"+key+"_" + str(args.variant) + "_b" + str(args.batch_size) + "_f" + str(args.num_filters) + "_l" + str(args.max_len)+".csv", index=False)
        
    else:
        args.time_hots = 0
        args.time_fp = 0
        args.time_fp2 = 0
        args.time_fp_all = 0
        args.time_bp = 0
        args.device = 'cpu'
        mpath = ("../model/"
                 + str(args.variant)
                 + "_fold" + str(0)
                 + "_b" + str(args.batch_size)
                 + "_f" + str(args.num_filters)
                 + "_l" + str(args.max_len))
        args = util.prepare_data(args, fold=0)
        args.loss_fn = nn.CrossEntropyLoss(weight=args.class_weights, reduction='mean').to(args.device)
        malconv = get_malconv_variant(args)
        malconv.load_state_dict(torch.load(mpath)) #, map_location=torch.device(device=args.device)))
        vbatch_times, epoch_stats, cpu_tst_res= validate([], epoch_stats={}, args=args, ghandle=ghandle, mode='test', model=malconv)

        exetime = time.time() - st
        cpu_load = psutil.getloadavg()[2]
        CPUusage = cpu_load / psutil.cpu_count()
        GPUusage = 0
        #cfp = carbon_footprint()
        #cfp = cfp.get_carbonEmissions(actual_runTime_hours=(exetime / (60 * 60)), usageGPU_used=GPUusage, usageCPU_used=CPUusage)
        # print("Carbon Footprint (Emission)", cfp / 1000, "kgCO2e")
        #print("Execution Time:", round(exetime/60, 3), "mins", "\tCPULoadAvg [15m]:", cpu_load, "\tCPUusage", CPUusage, "\tGPUusage:", GPUusage, "\tCarbon:", cfp / 1000, "kgCO2e")

        cvres['res']['cpu_tst'] = cpu_tst_res
        cpu_tst_res['carbon'] = cfp
        cpu_tst_res['total_cpu_eval_time_mins'] = exetime/60
        pd.DataFrame.from_dict(data=[cvres['res']['cpu_tst']]).to_csv("../out/cpu/test_" + str(args.variant) + "_b" + str(args.batch_size) + "_f" + str(args.num_filters) + "_l" + str(args.max_len)+".csv", index=False)

    util.display_args()


if __name__ == "__main__":
    util = Util()
    util.set_seed(seed=42)
    args, conf = util.get_parameters()
    
    ghandle = GPU_Usage(args)
    ghandle.set_init_usage("scrub")
    
    if 'cpu' not in args.device:
        torch.cuda.set_device(int(args.device[-1]))
    # ##########################################################
    # Dummy variable to trigger the init CUDA context memory use
    torch.zeros((1,), device=args.device)
    # ##########################################################
    
    st = time.time()
    ghandle.set_init_usage("cuda_context")

    grid_run(args, ghandle)
