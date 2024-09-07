### HotConv - Malware Detection/Classification with Low Carbon Footprints


A low GPU memory footprint and energy-efficient Green-AI learning strategy for training the class of 1D CNNs with temporal-max pooling layer, such as the state-of-the-art malware detection CNN, MalConv. 
HotConv reduces the GPU memory footprint by harnessing the sparsity of relevant activations and gradients at the temporal max pooling layer without trading off model capacity. 

<p align="center">
<img src="https://github.com/Anonymous-conference-202x/HotConv/blob/main/hotconv.gif" width="400" align="center"/>

</p>

<video src="https://github.com/user-attachments/assets/fce4db8e-8cd5-4a63-9e6a-2bc3970e9528" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px">

  </video>
<!-- ![hotconv.gif](hotconv.gif) -->

A comparison of HotConv, HotConv-$_{Eco}$ (a variant of HotConv) and MalConv2 (the existing memory-efficient version of MalConv) is provided below:

<p align="center">
<img src="https://github.com/Anonymous-conference-202x/HotConv/blob/main/comparison.png?raw=true" width="400" align="center"/>
</p>

A comparison of approaches for loading very large training sequence data in existing and proposed methods.

<!-- ![dataloader.gif](dataloader.gif) -->

<video src="https://private-user-images.githubusercontent.com/43296253/365400445-3abe2fe3-83df-424f-80a1-91103b13d5bc.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU3NDMxMDksIm5iZiI6MTcyNTc0MjgwOSwicGF0aCI6Ii80MzI5NjI1My8zNjU0MDA0NDUtM2FiZTJmZTMtODNkZi00MjRmLTgwYTEtOTExMDNiMTNkNWJjLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTA3VDIxMDAwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNkOWZiZGY3NDJjNzA2ZjcwNGU5MWM0NTdjNjRmOTU2Yjc1Mzg5MzdhZDMwYzFjMjRmNTNhYWJjYzlmMDk4ZTkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.z7NtUt7obtYhFtw1v90HtcJoK9KcXifGgzVMmRnApwU" data-canonical-src="https://private-user-images.githubusercontent.com/43296253/365400445-3abe2fe3-83df-424f-80a1-91103b13d5bc.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU3NDMxMDksIm5iZiI6MTcyNTc0MjgwOSwicGF0aCI6Ii80MzI5NjI1My8zNjU0MDA0NDUtM2FiZTJmZTMtODNkZi00MjRmLTgwYTEtOTExMDNiMTNkNWJjLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTA3VDIxMDAwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNkOWZiZGY3NDJjNzA2ZjcwNGU5MWM0NTdjNjRmOTU2Yjc1Mzg5MzdhZDMwYzFjMjRmNTNhYWJjYzlmMDk4ZTkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.z7NtUt7obtYhFtw1v90HtcJoK9KcXifGgzVMmRnApwU" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px">

  </video>

#### Package Requirements:

* Python-3.11
* CUDA-12.3



```
Package            Version
---------------------------
numpy              1.23.4
nvidia-ml-py3      7.352.0
pandas             1.5.1
pip                22.3
psutil             5.9.4
PyYAML             6.0.1
scikit-learn       1.3.2
scipy              1.9.3
torch              1.13.0
torchsummary       1.5.1
torchvision        0.2.2.post3
tqdm               4.66.1
```


  
#### Datasets:
* [BODMAS](https://whyisyoung.github.io/BODMAS/) - Yang et al.
* [VirusTotal](https://www.virustotal.com/gui/home/)


* These datasets contain raw binaries and require tens to hundreds of gigabytes of storage. Hence not included as part of the code submission.
* Please download using the provided links (may need to request download access) and peform the below step to run the code.
* Place the binaries for the public BODMAS dataset in a `foo` directory, and 
configure the `foo/` directory (notice the '/' at the end) as the `binaries_location` in `config/config.yaml`.




#### Directory Setup:
For evaluation purposes, we provide separate GPU and RAM implementations.

* **impl** - Implementation addressing GPU bottleneck for MalConv, MalConv2 and proposed approaches (HotConv and HotConv$_{Eco}$). 
Raff et al.'s implementation of MalConv2 with extended auxiliary architecture, which we refer to as MalConv2+ in our work,
can be found at the [link](https://github.com/NeuromorphicComputationResearchProgram/MalConv2).



```buildoutcfg
HotConv
    ├───impl                
    │   ├───data                # Contains csv list of samples for train/valid/test and partitions of the dataset in pickle format
    │   ├───model               # location to save trained models
    │   ├───out                 
    │   └───src                 
    │       ├───actions         
    │       ├───config          # parameter setting to train code
    │       ├───malconv2        # Adaptation of MalConv2 without the auxiliary architecture
    │       ├───model           # Model architecture - instantiation and training logic
    │       └───utilz           # Data Loader
```


#### Settings:

* Naming convention for cross-validation train/val/test csv 

(Use fold=0 for time-stamp based split)
```
train_bodmas_f94_fold_<FOLD INDEX>.csv
val_bodmas_f94_fold_<FOLD INDEX>.csv
test_bodmas_f94_fold_<FOLD INDEX>.csv
```
* Column Format: `FileName, Label, File_Size`. 
* (1) For standard training: Files can be in any order, and keep shuffle=True in data loader (utilz/utilz.py).

(2) For collecting memory metrics: The files are expected to be listed in a descending order of file size (as followed by Raff et al. for MalConv2) - keep shuffle=False in loader.
Reason: To get a stable reading from nvidia-smi by hitting the model first with a mini-batch of the largest samples.

* `src/config/config.yaml` - contains the configuration parameters.
The parameters are initialized with the setting to reproduce the memory reduction reported in the paper - 
our approach uses 1/22 of the $GRMM_{peak}$ memory consumed by MalConv2.

* Configure the root path containing the above cross-validation files of the datasets in `config.yaml` under `dpath`. Example: `../data/`.
* Configure the child directory containing the train/val/test csv of the desired dataset as `dataset`. Example: To point to `../data/bodmas_f94` just mention `bodmas_f94`.
* `seed` is enabled in `run.py`



#### Code Execution:

* Ensure no other program occupies GPU until HotConv execution ends - to get a clean reading of memory usage.
As our results depend on nvidia-smi to report model memory consumption.

* Set the model name to run in the `variant` field in config/config.yaml, 
and run the code in `src` directory with `python run.py` or python3 as needed.

* Average time for completing an epoch in BODMAS dataset with the setting used in paper is approx. 20 to 40 minutes, depending on GPU model.
Training time may reduce when dedicated GPU/RAM/hard disk is used instead of using specific resources in a cluster setting.


#### Interpreting Results:

* The memory usage metric - $GRMM_{peak}$ (defined in the paper) is configured to be printed for the first 5 batches in an epoch at the end of optimization step during training, and once at the end of validation phase, such that overall memory usage can be observed.

### Common Errors:

* The PyTorch's unpool operation has an open issue that it may throw "ValueError: invalid output_size "(XXX,)" (dim 0 must be between 0 and YYY)", when a very low window size (or stride) is used from the current setting. 
  To mitigate, reduce the FP slice size (in powers of 2).
  
* Note that the submitted code works if exact package requirements mentioned above are met (tested on NVIDIA RTX 6000 Ada).  
Some examples of runtime errors when using different setup are given below (tested on NVIDIA GeForce 1050 and RTX 4090 with lower CUDA versions):

* For errors on loading a trained model for evaluation, disable map_location attribute for load_state_dict method:

```model.load_state_dict(torch.load(mpath)) #, map_location=torch.device(device=args.device)))```


* For below error, batch size may need to be reduced. 

```RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle) ```

