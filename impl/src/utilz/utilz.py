import sys
import yaml
import numpy as np
import pandas as pd
from utilz.preprocess_malconv import SeqDataset, write_pred, genData
from torch.utils.data import DataLoader
import torch
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import pickle
import torch
import tqdm as tqdm
import torch.nn.functional as F


class Util():
    def __init__(self, ):
        super(Util, self).__init__()
        self.base_args = None

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def get_parameters(self, ):
         # Load config file for experiment
        try:
            config_path = "config/config.yaml"
            seed = 42
            conf = yaml.safe_load(open(config_path, 'r'))
        except Exception as e:
            print(str(e))
            sys.exit()
     
        # Parameter Settings   
        parser = argparse.ArgumentParser()
        parser.add_argument('--variant'                  , default=conf['variant'], help='')
        parser.add_argument('--dataset'                  , default=conf['dataset'], help='')
        # parser.add_argument('--use_gpu'                  , type=bool, default=True, help='')
        parser.add_argument('--device'                   , type=str, default='cuda:'+str(conf['gpu_id']) if conf['gpu_id']>=0 else 'cpu', help='')
        parser.add_argument('--learning_rate'            , type=float, default=conf['learning_rate'], help='')
        parser.add_argument('--early_stopping_patience'  , type=int, default=conf['early_stopping_patience'], help='')

        parser.add_argument('--batch_size'               , type=int, default=conf['batch_size'], help='')
        parser.add_argument('--window_size'              , type=int, default=conf['window_size'], help='How wide should the filter be')
        parser.add_argument('--stride'                   , type=int, default=conf['stride'], help='Filter Stride')
        parser.add_argument('--num_filters'              , type=int, default=conf['num_filters'], help='Total number of channels in output')
        parser.add_argument('--epochs'                   , type=int, default=conf['epochs'], help='How many training epochs to perform')
        # parser.add_argument('--subloader_batch_size'     , type=int, default=min(conf['batch_size'], conf['subloader_batch_size']), help='To load a large batch in parts')
        parser.add_argument('--max_len'                  , type=int, default=conf['max_len'],help='Maximum length of input file in bytes, at which point files will be truncated')
        
        parser.add_argument('--fp_slice_size'         , type=int, default=min(conf['max_len'], conf['fp_slice_size']), help='')
        # parser.add_argument('--fp2_slice_size'      , type=int, default=min(conf['window_size'], conf['fp2_slice_size']), help='')
        parser.add_argument('--fp2_slice_size'           , type=str
                            , default=conf['fp_slice_size'] if conf['fp2_slice_size'] == 'MIN' else conf['window_size'] * conf['num_filters']
                            , help='Min: window_size, MAX: num_filters * window_size')
        parser.add_argument('--dpath'                    , type=str, default=conf['dpath'], help='Dataset Path')
        parser.add_argument('--binaries_location', type=str, default=conf['binaries_location'], help='')
        # parser.add_argument('--folds'                    , type=int, default=conf['folds'], help='')
        parser.add_argument('--topk'                     , type=int, default=conf['topk'], help='topk accuracy')
        parser.add_argument('--num_workers'              , type=int, default=conf['num_workers'], help='for dataloader')
        parser.add_argument('--test_only_mode', type=bool, default=True if conf['gpu_id'] == -1 else False, help='No training performed when True. Tests model from fold 0 by default.')
        
        # parser.add_argument('--do_truncate'              , type=bool, default=conf['do_truncate'], help='Truncate files to max_len')
        # parser.add_argument('--batch_padding'            , type=bool, default=conf['batch_padding'], help='False: all samples padded to max_len. True: samples within a batch are padded to max sample size in the batch')
        
        args = parser.parse_args()
        args.early_stopping_patience_ = args.early_stopping_patience
        args.fp2_slice_size_ = conf['fp2_slice_size']
        args.folds = 1  # Set to 1 if following time-split based evaluation on dataset, and place the correponding time-split based train/val/test files. 
        args.do_truncate = True  # Truncate files to max_len. Current max_len is 512MB which is far greater than any file in the dataset
        args.batch_padding = True  # False: all samples padded to max_len. True: samples within a batch are padded to max sample size in the batch
        args.cpu_load_avg = np.array([])
        args.subloader_batch_size = args.batch_size # min(16, args.batch_size)
        self.base_args = args

        return args, conf
        
    def display_args(self,):
        print("\n Configuration Parameters:\n-------------------------------")
        dict_args = vars(self.base_args)
        keys = ["variant", "batch_size", "num_filters", "max_len", "device", "epochs", "fp_slice_size", "fp2_slice_size"]
        for arg in keys:
            print(arg, ":", dict_args[arg])

    def prepare_data(self, args, fold):
        try:
            args.train_files = pd.read_csv(args.dpath + args.dataset + "/train_"+args.dataset+"_fold_"+str(fold)+".csv", header=None).values
            args.valid_files = pd.read_csv(args.dpath + args.dataset + "/val_"+args.dataset+"_fold_"+str(fold)+".csv", header=None).values
            args.test_files = pd.read_csv(args.dpath + args.dataset + "/test_"+args.dataset+"_fold_"+str(fold)+".csv", header=None).values
        except Exception as e:
            print(str(e))
            print("\n\nSample list for the fold "+str(fold)+" not found. \n\nEXITING . . .")
            sys.exit()
        args.train_files = args.train_files[: len(args.train_files) // args.batch_size * args.batch_size]
        args.valid_files = args.valid_files[: len(args.valid_files) // args.batch_size * args.batch_size]
        args.test_files = args.test_files[: len(args.test_files) // args.batch_size * args.batch_size]
        
        args.train_population = len(args.train_files)
        args.valid_population = len(args.valid_files)
        args.test_population = len(args.test_files)
    
        for phase in ['train_corpus', 'val_corpus', 'test_corpus']:
            if not os.path.exists(args.dpath + args.dataset + "/" + phase + ".pkl"):
                corpus_path = self.picklify(args, phase)
            else:
                print("Loading existing partitions...")

            if 'train' in phase:
                args.train_corpus = self.load_pickled(args.dpath + args.dataset + "/" + phase + ".pkl")
            elif 'val' in phase:
                args.val_corpus = self.load_pickled(args.dpath + args.dataset + "/" + phase + ".pkl")
            else:
                args.test_corpus = None
                # args.test_corpus = self.load_pickled(args.dpath + args.dataset + "/" + phase + ".pkl")
    
        args.seqdataset_train = SeqDataset(args.train_files, args.max_len, args.window_size, args.fp_slice_size, args.do_truncate, batch_padding=args.batch_padding, binaries_location=args.binaries_location, batch_size=args.batch_size, corpus_ram=args.train_corpus)
        args.seqdataset_valid = SeqDataset(args.valid_files, args.max_len, args.window_size, args.fp_slice_size, args.do_truncate, batch_padding=args.batch_padding, binaries_location=args.binaries_location, batch_size=args.batch_size, corpus_ram=args.val_corpus)
        args.seqdataset_test = SeqDataset(args.test_files, args.max_len, args.window_size, args.fp_slice_size, args.do_truncate, batch_padding=args.batch_padding, binaries_location=args.binaries_location, batch_size=args.batch_size, corpus_ram=args.test_corpus)

        args.train_batch_collater = args.seqdataset_train.collater if args.batch_padding else None
        args.valid_batch_collater = args.seqdataset_valid.collater if args.batch_padding else None
        args.test_batch_collater = args.seqdataset_test.collater if args.batch_padding else None
        
        shuffle = True
        print("SHUFFLE :", shuffle)
        args.dataloader = DataLoader(args.seqdataset_train, batch_size=args.subloader_batch_size, shuffle=shuffle, num_workers=args.num_workers, persistent_workers=False #True
                                    , collate_fn=args.train_batch_collater
                                    , pin_memory=False)
        args.validloader = DataLoader(args.seqdataset_valid, batch_size=args.subloader_batch_size, shuffle=shuffle, num_workers=args.num_workers, persistent_workers=False #True
                                    , collate_fn=args.valid_batch_collater
                                    , pin_memory=False)
        args.testloader = DataLoader(args.seqdataset_test, batch_size=args.subloader_batch_size, shuffle=shuffle, num_workers=args.num_workers, persistent_workers=False #True
                                    , collate_fn=args.test_batch_collater
                                    , pin_memory=False)
    
        args.num_classes = len(np.unique(args.train_files[:, 1]))
        print("Train Files:", args.train_population, "\t\t", "Valid Files:", args.valid_population, "\t\t", "Test Files:", args.test_population
              , "\t#. classes train/val/test:", args.num_classes, len(np.unique(args.valid_files[:, 1])), len(np.unique(args.test_files[:, 1])), "\n")

        args.topk = args.topk if args.num_classes > 2 else 1

        args.class_weights = compute_class_weight('balanced', classes=np.unique(args.train_files[:, 1]), y=args.train_files[:, 1])
        args.class_weights_dict = {}
        for i in np.unique(args.train_files[:,1]): # np.unique(np.concatenate((args.train_files[:, 1], args.valid_files[:, 1], args.test_files[:, 1]))):
            try:
                args.class_weights_dict[i] = args.class_weights[i]
            except Exception as e:
                pass

        args.train_sample_weights = compute_sample_weight(class_weight=args.class_weights_dict, y=args.train_files[:, 1])
        args.valid_sample_weights = compute_sample_weight(class_weight=args.class_weights_dict, y=args.valid_files[:, 1])
        args.test_sample_weights = compute_sample_weight(class_weight=args.class_weights_dict, y=args.test_files[:, 1])
        args.class_weights = torch.tensor(args.class_weights,dtype=torch.float)

        return args


    def picklify(self, args, phase):
        if 'train' in phase:
            flist = args.train_files
        elif 'val' in phase:
            flist = args.valid_files
        else:
            flist = args.test_files
        print("Pickling", phase, ":", len(flist))
        corpus = {}
        tot_size = 0
        # LONG = np.longlong if int(np.__version__.split('.')[1]) > 20 else np.long
        flist = pd.DataFrame(flist, columns=['id', 'label', 'flen', 'family', 'time'])
        flist.sort_values(inplace=True, by='flen')
        flist = flist.values
        for id in tqdm.tqdm(range(len(flist))):
            _file, label, flen, *_ = flist[id]
            try:
               with open(args.binaries_location + _file, 'rb') as f:
                    x = f.read(min(args.max_len, flen))
                    x = np.frombuffer(x, dtype=np.uint8)
                    # x = np.concatenate([x, np.asarray([0] * (args.max_len - len(x)))])
                    # x = x.astype(np.int16)
                    #x = x.astype(LONG)
                    x = torch.tensor(x, dtype=torch.int16) + 1
                    
                    '''
                    if len(x) % args.fp_slice_size != 0:
                        residue = args.fp_slice_size - (len(x) % args.fp_slice_size)
                        x = F.pad(x, (0, residue), value=0)
                    '''
                    
                    # x = torch.tensor(x)
                    # label = torch.tensor([label])
                    corpus[_file] = x  #{'x': x, 'y': label}
            except Exception as e:
                print("Error in Picklify", str(e))
            

            fsize = os.stat(args.binaries_location + _file).st_size
            tot_size += fsize

            '''
            if id % 1000 == 0:
                # Intermediate Save pickled corpus
                corpus_path = args.dpath + args.dataset + "/" + phase + ".pkl"
                with open(corpus_path, "wb") as pt1handle:
                    pickle.dump(corpus, pt1handle)
            '''
        # Save pickled corpus
        corpus_path = args.dpath + args.dataset + "/" + phase + ".pkl"
        with open(corpus_path, "wb") as pt1handle:
            pickle.dump(corpus, pt1handle)

        return corpus_path
    

    def picklify2(self, args, phase):
        if 'train' in phase:
            flist = args.train_files
        elif 'val' in phase:
            flist = args.valid_files
        else:
            flist = args.test_files
        print("Pickling", phase, ":", len(flist))
        corpus = {}
        tot_size = 0
        LONG = np.longlong if int(np.__version__.split('.')[1]) > 20 else np.long
        flist = pd.DataFrame(flist, columns=['id', 'label', 'flen', 'family', 'time'])
        flist.sort_values(inplace=True, by='flen')
        flist = flist.values
        for id in tqdm.tqdm(range(len(flist))):
            _file, label, flen, *_ = flist[id]
            if flen > args.max_len:
                continue # corpus[_file] = None
            try:
               with open(args.binaries_location + _file, 'rb') as f:
                    x = f.read(min(args.max_len, flen))
                    x = np.frombuffer(x, dtype=np.uint8)
                    # x = np.concatenate([x, np.asarray([0] * (args.max_len - len(x)))])
                    # x = x.astype(np.int16)
                    #x = x.astype(LONG)
                    x = torch.tensor(x, dtype=torch.int16) + 1
                    # print(len(x))
                    
                    '''
                    if len(x) % args.fp_slice_size != 0:
                        residue = args.fp_slice_size - (len(x) % args.fp_slice_size)
                        x = F.pad(x, (0, residue), value=0)
                    '''

                    #x = torch.tensor(x)
                    # label = torch.tensor([label])
                    corpus[_file] = x  #{'x': x, 'y': label}
            except Exception as e:
                print("Error in Picklify", str(e))
            

            fsize = os.stat(args.binaries_location + _file).st_size
            tot_size += fsize

            '''
            if id % 1000 == 0:
                # Intermediate Save pickled corpus
                corpus_path = args.dpath + args.dataset + "/" + phase + ".pkl"
                with open(corpus_path, "wb") as pt1handle:
                    pickle.dump(corpus, pt1handle)
            '''
        # Save pickled corpus
        corpus_path = args.dpath + args.dataset + "/" + phase + ".pkl"
        with open(corpus_path, "wb") as pt1handle:
            pickle.dump(corpus, pt1handle)

        return corpus_path



    def load_pickled(self, corpus_path, max_len=None):
        with open(corpus_path, 'rb') as pkl:
            pObj = pickle.load(pkl)

        if max_len:
            for k in pObj.keys():
                pObj[k].data = pObj[k][max_len]

        return pObj
    
    def load_files_list(self, flist_path):
        with open(flist_path, 'rb') as pkl:
            pObj = pickle.load(pkl)

        if max_len:
            for k in pObj.keys():
                pObj[k].data = pObj[k][max_len]

        return pObj
