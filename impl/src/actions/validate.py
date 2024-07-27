import torch
from torch.autograd import Variable
import time
import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, top_k_accuracy_score, precision_recall_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import gc
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def validator(model, exe_input, args, ghandle):
    outputs = None
    if args.variant == 'proposed':
        outputs = model.forward_to_eval(exe_input, ghandle) if args.device != 'cpu' else model.forward_to_eval_cpu(exe_input, ghandle)
    elif args.variant == 'malconv2':
        outputs, _ = model(exe_input, ghandle, args)
    elif args.variant == 'malconv':
        outputs = model(exe_input.cuda())
    return outputs


def validate(batch_times, epoch_stats, args, ghandle, mode, model):
        eval_time = time.time()
        if mode == 'val':
            loader = args.validloader
            sample_weights = args.valid_sample_weights
            population = args.valid_population
        elif mode == 'test':
            loader = args.testloader
            sample_weights = args.test_sample_weights
            population = args.test_population

        model.device = args.device
        model.eval()
        eval_train_correct = 0
        eval_train_total = 0
        eval_train_loss = 0

        preds = []
        truths = []
        itr = 0

        with torch.no_grad():
            for batch_data in tqdm.tqdm(loader):
                if args.subloader_batch_size >= args.batch_size:
                    exe_input = batch_data[0]
                    label = batch_data[1].to(args.device)
                else:
                    if itr == 0:
                        exe_input = batch_data[0]
                        label = batch_data[1].to(args.device)
                        itr += 1
                        continue
                    else:
                        exe_input, label = args.seqdataset_valid.concat_var_len_batches(exe_input, label, batch_data, args.device, args.batch_padding)
                        itr += 1
                        if itr == args.batch_size / args.subloader_batch_size:
                            itr = 0
                        else:
                            continue
                
                vbt = time.time()
                vmx = exe_input.shape[1]
                
                if len(exe_input[0]) % args.window_size != 0:
                    residue = args.window_size - (len(exe_input[0]) % args.window_size)
                    exe_input = F.pad(exe_input, (0, residue), value=0)
                    
                label = label[:, 0]
                
                label = Variable(label.long(), requires_grad=False)  
                model.fp_slice_size = args.fp_slice_size
                outputs = validator(model, exe_input, args, ghandle)
                
                del exe_input, batch_data
                gc.collect()
                _, predicted = torch.max(outputs.data, 1)

                preds.extend(torch.softmax(outputs, dim=-1).data.detach().cpu().numpy())
                truths.extend(label.detach().cpu().numpy().ravel())
                
                eval_train_total += label.size(0)
                eval_train_correct += (predicted == label).sum().item()

                tloss = args.loss_fn(outputs, label)
                eval_train_loss += outputs.shape[0] * tloss.item()
                
                batch_times.append((time.time() - vbt, vmx, tloss.item()))                
                
        preds = torch.Tensor(preds)
        row_sums = torch.sum(preds, 1)
        preds = torch.div( preds , row_sums.unsqueeze(dim=1))
        
        predictions = np.argmax(preds, axis=1)
        # print(len(np.unique(truths)), len(np.unique(predictions)))
        if epoch_stats:
            epoch_stats[mode+'_auc'] = roc_auc_score(truths, predictions if args.num_classes==2 else preds, multi_class='ovr')
            epoch_stats[mode+'_loss'] = eval_train_loss / population
            epoch_stats[mode+'_acc'] = eval_train_correct * 1.0 / eval_train_total
        
        class_list = list(range(args.num_classes))
        topkacc = top_k_accuracy_score(truths, predictions if args.num_classes==2 else preds, k=1 if args.num_classes==2 else args.topk, labels=class_list, sample_weight=sample_weights)
        precision, recall, f1, _ = precision_recall_fscore_support(truths, predictions, labels=class_list, average='weighted', sample_weight=sample_weights)

        binarized_truths = label_binarize(truths, classes=class_list)
        prc_auc = average_precision_score(binarized_truths, preds, average="weighted", sample_weight=sample_weights)
        roc_auc = roc_auc_score(truths, preds, average="weighted", sample_weight=sample_weights, max_fpr=None, multi_class="ovr", labels=class_list)
        precision = round(precision, 6)
        recall = round(recall, 6)
        f1 = round(f1, 6)
        prc_auc = round(prc_auc, 6)
        roc_auc = round(roc_auc, 6)
        topkacc = round(topkacc, 6)
        print("weighted\tprecision:", precision, "   recall:", recall, "   f1:", f1, "   prc_auc:", prc_auc, "   roc_auc:", roc_auc)

        eval_time = (time.time() - eval_time) / 60
        res_data = {"variant": args.variant,
                    "batch_size": args.batch_size,
                    "num_filters": args.num_filters,
                    "max_len": args.max_len,
                    "mode": mode,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "topkacc": topkacc,
                    "prc_auc": prc_auc,
                    "roc_auc": roc_auc,
                    "eval_time": eval_time}
        
        return batch_times, epoch_stats, res_data


if __name__ == "__main__":
    validator()
