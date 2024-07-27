import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from torch import nn
import random
import psutil
import time


def write_pred(pred, file_path, idx):
    test_pred = [item for sublist in pred for item in sublist]
    with open(file_path, 'w') as f:
        for idx, pred in zip(idx, test_pred):
            print(idx.upper()+','+str(pred[0]), file=f)


def genData(good_dir, bad_dir, sort_by_size, population):
    all_files = []
    for root_dir, dirs, files in os.walk(good_dir):
        print("Total Benign Found:", len(files))
        for file in files[:int(population/2)]:
            to_add = os.path.join(root_dir, file)
            all_files.append((to_add, 0, os.path.getsize(to_add)))
        break

    for root_dir, dirs, files in os.walk(bad_dir):
        print("Total Malware Found:", len(files))
        for file in files[:int(population/2)]:
            to_add = os.path.join(root_dir, file)
            all_files.append((to_add, 1, os.path.getsize(to_add)))
        break

    print("Total Files:", len(all_files))
    if sort_by_size:
        all_files.sort(key=lambda filename: filename[2])
        print('sorted files')
    return all_files


class SeqDataset(Dataset):
    def __init__(self, all_files, max_len, window_size, fp_slice_size, truncate, batch_padding, binaries_location, batch_size, corpus_ram=None):
        self.all_files = all_files
        self.max_len = max_len
        self.truncate = truncate
        self.window_size = window_size
        self.fp_slice_size = fp_slice_size
        self.batch_padding = batch_padding
        self.bin_loc = binaries_location
        self.LONG = np.longlong if int(np.__version__.split('.')[1]) > 20 else np.long
        self.corpus_ram = corpus_ram
        self.template = torch.zeros((batch_size, self.max_len))

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__old(self, idx):
        bt = time.time()
        _file, y, flen, *_ = self.all_files[idx]
        try: 
            with open(self.bin_loc+_file, 'rb') as f:
                # x = f.read(flen)
                x = f.read(min(self.max_len, flen))
                x = np.frombuffer(x, dtype=np.uint8)
                x = torch.tensor(x, dtype=torch.int16) + 1

        except Exception as e:
            print("Data loading error - preprocess_malconv.py", str(e))
            print("This error could be related to Numpy version. Here, the version found is", np.__version__)
        #print("Load preprocess time:", time.time() - bt)
        return x, torch.as_tensor([y]) # torch.tensor(x), torch.tensor([y])

    def __getitem__(self, idx):
        _file, y, flen, *_ = self.all_files[idx]
        try: 
            try:
                x = self.corpus_ram[_file][:self.max_len]
            except Exception as e:
                print("Sample ", _file," not found in RAM", str(e))
                '''
                print("*******************************************  Read from disk only if not found in partition")
                with open(self.bin_loc+_file, 'rb') as f:
                    # x = f.read(flen)
                    x = f.read(min(self.max_len, flen))
                    x = np.frombuffer(x, dtype=np.uint8)
                    x = torch.tensor(x, dtype=torch.int16) + 1
                '''
        except Exception as e:
            print("Data loading error - preprocess_malconv.py", str(e))
            print("This error could be related to Numpy version. Here, the version found is", np.__version__)
        return x, torch.as_tensor([y]) # torch.tensor(x), torch.tensor([y])

    def collater_old(self, batch):
        bt = time.time()
        # random.shuffle(batch)
        vecs = [x[0] for x in batch]     
        # ct = time.time()
        vecs = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
        #print(vecs.shape, "collater")
        # print("===>", time.time() - ct)
        # x = torch.stack([torch.cat((x[0], self.template[x[0].shape[-1]:])) for x in batch])
        y = torch.stack([x[1] for x in batch])
        # x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
        # Below calc follows (total - avail / total) as recommended by psutil to find ram usage %
        # 512 is the RAM capacity
        # print('GPU IMPL - RAM_USAGE (GB):', (psutil.virtual_memory()[0]/1024**3) - (psutil.virtual_memory()[1]/1024**3) / (psutil.virtual_memory()[0]/1024**3) * 512) 
        #print("Padding time:", time.time() - bt)
        return vecs, y
    
    def collater(self, batch):
        x = self.template
        for i, sample in enumerate(batch):
            x[i][:len(sample[0])] = sample[0]

        #vecs = [x[0] for x in batch]     
        # ct = time.time()
        #vecs = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
        #print(vecs.shape, "collater")
        # print("===>", time.time() - ct)
        # x = torch.stack([torch.cat((x[0], self.template[x[0].shape[-1]:])) for x in batch])
        y = torch.stack([x[1] for x in batch])
        # x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
        # Below calc follows (total - avail / total) as recommended by psutil to find ram usage %
        # 512 is the RAM capacity
        # print('GPU IMPL - RAM_USAGE (GB):', (psutil.virtual_memory()[0]/1024**3) - (psutil.virtual_memory()[1]/1024**3) / (psutil.virtual_memory()[0]/1024**3) * 512) 
        return x, y
        end_idx = torch.nonzero(x, as_tuple=True)[1][-1].item()
        if end_idx % self.fp_slice_size != 0:
            residue = self.fp_slice_size - (end_idx % self.fp_slice_size)
            end_idx += residue
            #print(end_idx % self.fp_slice_size)
        return x[:, :end_idx], y
        
    def concat_var_len_batches(self, exe_input, label, batch_data, device, batch_padding=True):
        try:
            if batch_padding:
                new_max = max(exe_input.shape[-1], batch_data[0].shape[-1])
                if exe_input.shape[-1] < new_max:
                    exe_input = torch.cat((
                        nn.ConstantPad1d((0, new_max - exe_input.shape[-1]), 0)(exe_input),
                        batch_data[0]
                        ))
                else:
                    exe_input = torch.cat((
                        exe_input,
                        nn.ConstantPad1d((0, new_max - batch_data[0].shape[-1]), 0)(batch_data[0])))
                label = torch.cat((label.to(device), batch_data[1].to(device)))
            else:
                exe_input = torch.cat((exe_input, batch_data[0]))
                label = torch.cat((label, batch_data[1].to(device)))
        except Exception as e:
            print(str(e))
            exit() 
        return exe_input, label

