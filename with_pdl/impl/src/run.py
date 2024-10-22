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

    xt = time.time()
    loss.backward()
    args.time_bp += time.time() - xt
    
    xt = time.time()
    args.adam_optim.step()
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

    xt = time.time()
    batch_blocks = malconv.gather_hot_blocks(fcache, exe_input)
    args.time_hot += time.time() - xt
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

    xt = time.time()
    malconv.backprop_selective_gradient_attribution(args.adam_optim, fcache, batch_blocks, ghandle)
    args.time_conv_bp += time.time()-xt
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

        num_samples_processed = args.partition_size * 3 # 0    
        args.partition_size = 256                                  
        while num_samples_processed < len(args.train_files):
            flist = args.train_files[num_samples_processed:num_samples_processed + args.partition_size]
            args = util.setDataLoader(flist, args, phase='train')
            print("-----------> Partition : ", int(num_samples_processed / args.partition_size) + 1)
            for batch_data in tqdm.tqdm(args.dataloader):
                if args.subloader_batch_size >= args.batch_size:
                    exe_input = batch_data[0]
                    label = batch_data[1].to(args.device)

                exe_input = exe_input.long()
                if len(exe_input[0]) % args.fp_slice_size != 0:
                    residue = args.fp_slice_size - (len(exe_input[0]) % args.fp_slice_size)
                    exe_input = F.pad(exe_input, (0, residue), value=0)

                tbt = time.time()
                tmx = exe_input.shape[1]
                label = Variable(label[:, 0].long(), requires_grad=False)  # ***********************       bodmas  - 1
                args.adam_optim.zero_grad()
                
                lt = time.time() - lt
                args.time_load += lt

                args = train(malconv, exe_input, label, args, ghandle)
                ot = time.time()
                tbatch_times.append((time.time() - tbt, tmx, args.cur_batch_loss))
                if len(tbatch_times) <= 5:
                    ghandle.get_gpu_usage(args.variant + ": AFTER Optimization Step")
                args.time_others += time.time() - ot
                lt = time.time()
            num_samples_processed += len(flist)
        num_samples_processed = 0

        args.time_total = time.time() - tlt

        epoch_stats['tr_acc'] = args.train_correct * 1.0 / args.train_total
        # print(epoch_stats['tr_acc'], args.train_correct, args.train_total)
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
        vbatch_times, epoch_stats, val_res = validate(vbatch_times, epoch_stats, args, ghandle, utilObj=util, mode='val', model=malconv)
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

    tbatch_times, _, tst_res = validate(tbatch_times, None, args, ghandle, utilObj=util, mode='test', model=malconv)
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

        for key in cvres['res'].keys():
            cvres['res'][key]['avg_train_epoch_time'] = avg_train_epoch_time
            cvres['res'][key]['total_training_time'] = exetime / 60
            cvres['res'][key]['total_training_epochs'] = epoch
            #cvres['res'][key]['carbon'] = cfp
            cvres['res'][key]['grmmpeak'] = ghandle.get_grmmpeak_()
            cvres['res'][key]['gpu_total_usage'] = ghandle.get_total_usage_()
            cvres['res'][key]['gpu_usage_factor'] = 0 # GPUusage
            cvres['res'][key]['cpu_usage_factor'] = 0 # CPUusage
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
