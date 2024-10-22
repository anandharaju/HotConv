import torch
import torch.nn as nn
import time
from itertools import starmap, product
from multiprocessing.pool import Pool
import pandas as pd
import numpy as np
from collections import Counter

    
class MalConv_Proposed(nn.Module):
    def __init__(self, args):
        super(MalConv_Proposed, self).__init__()
        self.device = args.device
        self.max_len = args.max_len
        self.window_size = args.window_size
        self.batch_size = args.batch_size
        self.num_filters = args.num_filters
        self.embd_dim = 8

        # ##############################################################
        # To use same embedding weights for all models,
        # use the _weight parameter to supply the custom weights.
        # Shape of weights: (257, 8)
        # ##############################################################
        # CUSTOM_WEIGHT_INIT = torch.nn.init.normal(torch.empty(257, 8)))
        # self.embd = nn.Embedding(257, self.embd_dim, padding_idx=0, _weight=CUSTOM_WEIGHT_INIT)
        self.embd = nn.Embedding(257, self.embd_dim, padding_idx=0)

        self.stride = args.stride
        self.conv_1 = nn.Conv1d(self.embd_dim, args.num_filters, args.window_size, stride=args.stride, bias=True)
        self.conv_2 = nn.Conv1d(self.embd_dim, args.num_filters, args.window_size, stride=args.stride, bias=True)
        self.pooling = nn.AdaptiveMaxPool1d(1, return_indices=True)
        self.unpool = nn.MaxUnpool1d(args.num_filters, stride=None, padding=0)
        self.fc_1 = nn.Linear(args.num_filters, args.num_filters)
        self.fc_2 = nn.Linear(args.num_filters, args.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky_relu= nn.LeakyReLU()
        self.fp_slice_size = args.fp_slice_size
        self.fp2_slice_size = args.fp2_slice_size
        self.num_classes = args.num_classes
        
        self.top_activs = torch.zeros((self.batch_size, self.num_filters, 1), device=self.device)
        self.top_slices_index = torch.zeros((self.batch_size, self.num_filters, 1), dtype=int, device=self.device)
        self.ones = torch.ones((self.batch_size, self.num_filters, 1), dtype=int, device=self.device)

        #if self.fp2_slice_size != self.window_size:
        #    self.var = torch.ones((self.batch_size * 1, self.num_filters, self.num_filters), dtype=int, device=self.device)
        
        self.cnn2r = torch.zeros((self.batch_size, self.num_filters), device=self.device)
        self.relur = torch.zeros((self.batch_size, self.num_filters), device=self.device)
        self.sigmr = torch.zeros((self.batch_size, self.num_filters), device=self.device)
        self.final_top_activs = None
        
        
    # ##############################################################
    # Description: Used for the FP phase
    # This method collects the top-activations and their indices
    # for the given input by processing it slice-by-slice
    # ##############################################################
    def forward(self, x, ghandle=None):
        start = 0
        tracker = dict()

        self.top_activs.fill_(0)
        self.top_slices_index.fill_(0)
        self.cnn2r.fill_(0)
        self.relur.fill_(0)
        self.sigmr.fill_(0)

        tracker['cnn2r'] = self.cnn2r
        tracker['relur'] = self.relur
        tracker['sigmr'] = self.sigmr

        # ghandle.get_gpu_usage("a")
        
        while start < x.shape[1]:
            xprime_emb = self.embd(x[:, start: start + self.fp_slice_size].to(self.device))
            xprime_emb = torch.transpose(xprime_emb, -1, -2)  
            cnn1 = self.conv_1(xprime_emb)
            cnn2 = self.conv_2(xprime_emb)
            tracker['cnn2'] = cnn2
            cnn1 = self.relu(cnn1)
            # tracker['cnn1_relu'] = cnn1   # cnn1 contains relu output
            cnn2 = torch.sigmoid(cnn2)
            # tracker['cnn2_sigm'] = cnn2   #  cnn2 contains sigmoid output
            slice_grp_activs = cnn1 * cnn2
            
            slice_grp_activs, slice_grp_indices = self.pooling(slice_grp_activs)
            bool_mask = self.unpool(slice_grp_activs, slice_grp_indices, output_size=(self.batch_size, self.num_filters, int(self.fp_slice_size/self.window_size))).bool()
            
            new_max_indices = torch.where(slice_grp_activs > self.top_activs, True, False)
            self.top_activs = torch.where(new_max_indices, slice_grp_activs, self.top_activs)
            slice_grp_indices = self.ones * start + (slice_grp_indices * self.stride) if self.fp_slice_size != self.window_size else self.ones * start
            self.top_slices_index = torch.where(new_max_indices, slice_grp_indices, self.top_slices_index)

            new_max_indices = new_max_indices.view(-1, self.num_filters)
            #tracker['cnn2'] = torch.where(bool_mask, tracker['cnn2'], bool_mask)
            tracker['cnn2r'] = torch.where(new_max_indices, 
                                           torch.sum(torch.where(bool_mask, tracker['cnn2'], bool_mask), dim=2),
                                           # torch.sum(bool_mask * tracker['cnn2'], dim=2), 
                                           tracker['cnn2r'])
            #tracker.pop('cnn2')
            #tracker['cnn1_relu'] = torch.where(bool_mask, tracker['cnn1_relu'], bool_mask)
            tracker['relur'] = torch.where(new_max_indices, 
                                           torch.sum(torch.where(bool_mask, cnn1, bool_mask), dim=2), 
                                           # torch.sum(bool_mask * cnn1, dim=2), 
                                           tracker['relur'])
            #tracker.pop('cnn1_relu')
            #tracker['cnn2_sigm'] = torch.where(bool_mask, tracker['cnn2_sigm'], bool_mask)
            tracker['sigmr'] = torch.where(new_max_indices, 
                                           torch.sum(torch.where(bool_mask, cnn2, bool_mask), dim=2), 
                                           # torch.sum(bool_mask * cnn2, dim=2), 
                                           tracker['sigmr'])
            #tracker.pop('cnn2_sigm')

            start += self.fp_slice_size
        # ghandle.get_gpu_usage("b")
        tracker['top_activs'] = self.top_activs.view(-1, self.num_filters)
        tracker['top_slices_index'] = self.top_slices_index.view(-1, self.num_filters)

        self.final_top_activs = self.top_activs.view(-1, self.num_filters)
        self.final_top_activs.requires_grad = True
        self.final_top_activs.register_hook(self.hook_dense_to_pooling)
        x = self.fc_1(self.final_top_activs)
        x = self.leaky_relu(x)
        x = self.fc_2(x)

        return x, tracker

    # ##############################################################
    # Description: Used for the evaluation phase
    # computes the output for detection or classification
    # ##############################################################
    def forward_to_eval(self, x, ghandle):
        start = 0
        self.top_activs.fill_(0)
        while start < x.shape[1]:
            xprime_emb = self.embd(x[:, start: start + self.fp_slice_size].long().to(self.device, non_blocking=True))
            xprime_emb = torch.transpose(xprime_emb, -1, -2)
            # xprime_emb = xprime_emb.permute(0,2,1).contiguous()
            
            cnn1 = self.conv_1(xprime_emb)
            cnn2 = self.conv_2(xprime_emb)
            del xprime_emb
            cnn1 = self.relu(cnn1)
            cnn2 = torch.sigmoid(cnn2)
            slice_grp_activs = cnn1 * cnn2

            if self.fp_slice_size != self.window_size:
                slice_grp_activs, _ = self.pooling(slice_grp_activs)

            max_indices = torch.where(slice_grp_activs > self.top_activs, True, False)
            self.top_activs = torch.where(max_indices, slice_grp_activs, self.top_activs)
            start += self.fp_slice_size
            del cnn1, cnn2, slice_grp_activs, max_indices

        x = self.fc_1(self.top_activs.view(-1, self.num_filters))
        x = self.leaky_relu(x)
        x = self.fc_2(x)
        return x
        
    def forward_to_eval_cpu(self, x, ghandle):
        start = 0
        self.top_activs.fill_(0)
        while start < x.shape[1]:
            xprime_emb = self.embd(x[:, start: start + self.fp_slice_size].long().to(self.device, non_blocking=True))
            # xprime_emb = torch.transpose(xprime_emb, -1, -2)
            xprime_emb = xprime_emb.permute(0,2,1).contiguous()
            
            cnn1 = self.conv_1(xprime_emb)
            cnn2 = self.conv_2(xprime_emb)
            del xprime_emb
            cnn1 = self.relu(cnn1)
            cnn2 = torch.sigmoid(cnn2)
            slice_grp_activs = cnn1 * cnn2

            if self.fp_slice_size != self.window_size:
                slice_grp_activs, _ = self.pooling(slice_grp_activs)

            max_indices = torch.where(slice_grp_activs > self.top_activs, True, False)
            self.top_activs = torch.where(max_indices, slice_grp_activs, self.top_activs)
            start += self.fp_slice_size
            del cnn1, cnn2, slice_grp_activs, max_indices

        x = self.fc_1(self.top_activs.view(-1, self.num_filters))
        x = self.leaky_relu(x)
        x = self.fc_2(x)
        return x

    # **************************************
    # ************ Gradients ***************
    # **************************************

    # ##############################################################
    # Description: Gradient computation for Multiply layer
    # ##############################################################
    def gradient_multiply(self, inp1, inp2, grad):
        return torch.mul(inp2, grad), torch.mul(inp1, grad)

    # ##############################################################
    # Description: Gradient computation for Sigmoid layer
    # ##############################################################
    def gradient_sigmoid(self, inp, grad):
        inp = self.sigmoid(inp)
        inp = torch.mul(inp, 1. - inp)
        return torch.mul(inp, grad)

    # ##############################################################
    # Description: Gradient computation for ReLU layer
    # ##############################################################
    def gradient_relu(self, inp, grad):
        return grad
        # We can re-use the incoming gradients as ReLU grads.
        # Because the local gradient of ReLU is 1 only for x > 0.
        # - For top-activations, their local gradient is 1. so the incoming gradients is passed as-is to its location.
        # - For sub-maximal activations, their local gradient will be 1 but the gradients from pool/mult layer will be 0
        # Overall, the incoming gradients is back propagated as-is to next layer.

    # ##############################################################
    # Description: Gradient computation for convolution layer
    # filter weights
    # ##############################################################
    def gradient_sparse_convolutions_weight(self, x, grad1, grad2, ghandle):
        zt = time.time()
        # print(x.shape, grad1.shape, grad2.shape)
        grad1 = grad1.view(self.batch_size, self.num_filters, 1, 1).to(self.device)
        grad2 = grad2.view(self.batch_size, self.num_filters, 1, 1).to(self.device)
        res1 = torch.zeros((self.num_filters, self.embd_dim, self.window_size), device=self.device)
        res2 = torch.zeros((self.num_filters, self.embd_dim, self.window_size), device=self.device)
        # print("a) time:", time.time()-zt)
        # ghandle.get_gpu_usage("1")
        zt = time.time()
        # fstep defines the level of parallelism.
        fstep = int(self.fp2_slice_size / self.window_size) # Num. of filters' data to process at a time ftep % self.num_filters should be 0
        for f in range(0, self.num_filters, fstep):
            #print("\n\nb 1) time:", time.time()-zt)
            # cur_slice_embd = x[:, :, f*self.window_size:(f+fstep)*self.window_size].to(self.device).reshape(self.batch_size, self.embd_dim, fstep, self.window_size)
            cur_slice_embd = x[:, :, f*self.window_size:(f+fstep)*self.window_size]
            #print("b 1 1) time:", time.time()-zt)
            #ghandle.get_gpu_usage("2")
            if self.fp2_slice_size != (self.num_filters * self.window_size):
                cur_slice_embd = cur_slice_embd.to(self.device)
            #ghandle.get_gpu_usage("3")
            #print("b 1 2) time:", time.time()-zt)
            cur_slice_embd = cur_slice_embd.reshape(self.batch_size, self.embd_dim, fstep, self.window_size)
            #print("b 1 3) time:", time.time()-zt)
            #ghandle.get_gpu_usage("4")
            #print("b 2) time:", time.time()-zt)
            res1[f:(f+fstep), :, :] += torch.sum(torch.moveaxis((cur_slice_embd * torch.moveaxis(grad1[:, f:(f+fstep)], 2,1)),2,1), dim=0)
            #print("b 3) time:", time.time()-zt)
            res2[f:(f+fstep), :, :] += torch.sum(torch.moveaxis((cur_slice_embd * torch.moveaxis(grad2[:, f:(f+fstep)], 2,1)),2,1), dim=0)
            #print("b 4) time:", time.time()-zt)
        # print("Loop time:", time.time()-zt)
        
        zt = time.time()
        res1[f:(f+fstep), :, :] /= self.batch_size
        res2[f:(f+fstep), :, :] /= self.batch_size
        # print("c) time:", time.time()-zt)
        return res1.to(self.device), res2.to(self.device)

    # ##############################################################
    # Description: Gradient computation for Convolution layer input
    # ##############################################################
    def gradient_sparse_convolutions_input_v3(self, bse, grad1, grad2, ghandle):
        # No need for rotation as the gradients are scalar w.r.t a block
        # (grad_relu_to_conv1[b,f,a] * malconv.conv_1.weight[f].rot90(2)).rot90(2)
        for b in range(self.batch_size):
            # Each (gradient * filter) is linked to a different hot slice in the same input sequence
            temp1 = torch.mul(self.conv_1.weight, grad1[b].view(self.num_filters, 1, 1))
            temp2 = torch.mul(self.conv_2.weight, grad2[b].view(self.num_filters, 1, 1))
            bse[b] = (temp1 + temp2).view(8, self.num_filters * self.window_size)
        return bse
    
    def get_block_from_input(self, sample_index, top_block_index):
        if self.block_cache and sample_index in self.block_cache.keys() and top_block_index in self.block_cache[sample_index].keys():
            return self.block_cache[sample_index][top_block_index]
        else:
            block_data = self.exe_input[sample_index, top_block_index:(top_block_index + self.window_size)]
            if self.counters[sample_index][top_block_index] > 2:
                self.block_cache[sample_index] = {top_block_index: block_data}
                #print(self.block_cache.keys())
        return block_data
    
    # ##############################################################
    # Description: Helper method to collect Hot Delta Blocks
    # ##############################################################
    def gather_hot_blocks(self, tracker, exe_input):
        # bt = time.time()
        top_blocks_index = tracker['top_slices_index'].cpu()
        # Process the gathered slices in portions of size = window size or num.filters * window size.
        # It is beneficial to NOT store unique blocks, when processing extremely large sequences - where unique block count will be same as number of filters
        # Below code can be easily modified to store unique blocks when processing smaller sequeces - where unique block count will be far lesser than num. of filters,
        # as same blocks will be picked by different filters from within the same small sequence

        batch_blocks_bytes = torch.stack([torch.cat([exe_input[b, t:(t + self.window_size)] for t in top_blocks_index[b]]) for b in range(self.batch_size)])
        
        if self.fp2_slice_size != (self.num_filters * self.window_size):
            self.embd.cpu()
            batch_blocks = self.embd(batch_blocks_bytes).transpose(-1, -2)  # .permute(0,2,1).contiguous()
            self.embd.cuda()
        else:
            batch_blocks_bytes = batch_blocks_bytes.to(self.device)
            batch_blocks = self.embd(batch_blocks_bytes).transpose(-1, -2)  # .permute(0,2,1).contiguous()

        return batch_blocks
            
    # **************************************
    # ************** HOOK *****************
    # **************************************
    def hook_dense_to_pooling(self, grad):
        self.grads_dense_to_pooling = grad

    def get_gradients_dense_to_pooling(self):
        return self.grads_dense_to_pooling

    # ****************************************************
    # GRADIENT ATTRIBUTION-BASED BACKPROPAGATION
    # ****************************************************
    
    def backprop_selective_gradient_attribution(self, optim, fcache, batch_slices, ghandle=None):
        #yt = time.time()
        grad_dense_to_pool = self.get_gradients_dense_to_pooling()
        #print("\n\n",grad_dense_to_pool.shape)
        grad_mult_to_relur, grad_mult_to_sigmr = self.gradient_multiply(fcache['relur'], fcache['sigmr'], grad_dense_to_pool)
        #print(grad_mult_to_relur.shape, grad_mult_to_sigmr.shape)
        grad_sigm_to_conv2r = self.gradient_sigmoid(fcache['cnn2r'], grad_mult_to_sigmr)
        #print(grad_sigm_to_conv2r.shape)
        grad_conv1_weights, grad_conv2_weights = self.gradient_sparse_convolutions_weight(batch_slices, grad_mult_to_relur, grad_sigm_to_conv2r, ghandle)
        #print("Conv BP time:", time.time()-yt)
        
        # yt = time.time()
        self.conv_1.weight.requires_grad = True
        self.conv_2.weight.requires_grad = True
        self.conv_1.bias.requires_grad = True
        self.conv_2.bias.requires_grad = True

        self.conv_1.weight.grad = grad_conv1_weights
        self.conv_2.weight.grad = grad_conv2_weights
        self.conv_1.bias.grad = torch.sum(grad_mult_to_relur, dim=0)
        self.conv_2.bias.grad = torch.sum(grad_sigm_to_conv2r, dim=0)

        optim.step()

        self.conv_1.weight.requires_grad = False
        self.conv_2.weight.requires_grad = False
        self.conv_1.bias.requires_grad = False
        self.conv_2.bias.requires_grad = False
        # print("Opt time:", time.time()-yt)
