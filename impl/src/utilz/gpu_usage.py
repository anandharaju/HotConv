import nvidia_smi
import torch


class GPU_Usage():
    def __init__(self, args):
        super(GPU_Usage, self).__init__()
        self.DEVICE_ID = int(args.device[-1]) if args.device != "cpu" else 0
        self.init_usage = 0
        self.reset_usage = 0
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.DEVICE_ID)
        self.scrub = 0
        self.usage = 0
        self.total = 0

    def set_init_usage(self, usetype=None):
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        if usetype == "cuda_context":
            self.init_usage = int(mem_res.used / (1024 ** 2)) - self.scrub
            print(">>>>>>>>>>>>>>\tGPU Usage:\t\tGRMM_peak : " + str(self.init_usage) + " MB\t[ CUDA CONTEXT MEMORY ]")
            print("[ Note ] The above usage is for: CUDA context memory / OS services / background services / Other programs running in GPU. ")
            print("This usage could vary from hundreds of MBs to GBs due to several reasons. Please see the related notes in the paper for details.")
            print("Exclusion of the above memory also helps to align with the model memory estimation obtained via PyTorch's torchsummary.")
            print("[ Note ] Below we account for the memory consumption of the model from it's intantiation stage to runtime usage.")
        else:
            self.scrub = int(mem_res.used / (1024 ** 2))
            print(">>>>>>>>>>>>>>\tGPU Usage:\t\tGRMM_peak : " + str(self.scrub) + " MB\t[ GPU USAGE AT CODE START ]")
            print("\nNon-zero usage can be due to mem for GPU driver initialization, scrubbing ECC-enabled GPU, etc.")
            print("This may not show up in nvidia-smi command line.\n")
        self.total = mem_res.total / (1024**2)  # GPU capacity

    def resetter(self, reset_usage):
        print("Resetting . . .")
        self.reset_usage=reset_usage

    def get_gpu_usage(self, desc):
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        usage_in_MB = int(mem_res.used / (1024**2) - self.init_usage - self.scrub)
        print(">>>>>>>>>>>>>>\tGPU Usage:\t\tGRMM_peak : " + str(usage_in_MB) + " MB\t[ "+desc+" ]")
        self.usage = usage_in_MB
        return usage_in_MB
        
    def get_gpu_usage_factor_(self):
        return self.usage / self.total

    def get_init_usage_(self):
        return self.init_usage

    def get_grmmpeak_(self):
        return self.usage

    def get_total_usage_(self):
        return self.init_usage + self.usage


if __name__ == "__main__":
    gpu_handle = GPU_Usage(0)
    print(gpu_handle.get_gpu_usage("TRIAL"))
