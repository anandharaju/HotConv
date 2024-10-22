import numpy as np
import pandas as pd

class carbon_footprint:
    def __init__(self):
        self.server_location_carbonIntensity = 475
        self.memory_capacity = 48
        self.n_CPUcores = 36
        self.CPUpower = 8.33
        self.n_GPUs = 1
        self.GPUpower = 300
        self.PUE_used = 1.67
        self.PSF_used = 1
        # self.usageCPU_used = 0
        self.memoryPower=0.3725

    def get_carbonEmissions(self, actual_runTime_hours=24, usageGPU_used=1, usageCPU_used=0):
        runTime = actual_runTime_hours
        powerNeeded_CPU = self.PUE_used * self.n_CPUcores * self.CPUpower * usageCPU_used
        powerNeeded_GPU = self.PUE_used * self.n_GPUs * self.GPUpower * usageGPU_used
        # Power needed, in Watt
        powerNeeded_core = powerNeeded_CPU + powerNeeded_GPU
        powerNeeded_memory = self.PUE_used * (self.memory_capacity * self.memoryPower)
        powerNeeded = powerNeeded_core + powerNeeded_memory
        # Energy needed, in kWh (so dividing by 1000 to convert to kW)
        energyNeeded = runTime * powerNeeded * self.PSF_used / 1000
        carbonEmissions = energyNeeded * self.server_location_carbonIntensity
        return carbonEmissions

