import os
import time
import random
import shutil
import numpy as np
import ioh
from ioh import ProblemClass, Experiment
import matlab.engine

future=matlab.engine.start_matlab(background=True)
eng=future.result()  # Wait for MATLAB engine to start

class Opts:
    def __init__(self,dim,Instance,Index,Name,ShowIts,MaxEvals,MaxIts,Ftarget):
        self.dimension=dim
        self.instance=Instance
        self.index=Index
        self.name=Name
        self.maxevals=MaxEvals
        self.showits=ShowIts
        self.maxits=MaxIts
        self.ftarget=Ftarget  
        
    def to_dict(self):
        return {
            'dimension': self.dimension,
            'instance': self.instance,
            'index': self.index,
            'name': self.name,
            'maxevals': self.maxevals,
            'maxits': self.maxits,
            'showits': self.showits,
            'ftarget': self.ftarget
        }

class Algorithm:
    def __init__(self):
        self.x=0
        self.evals=1
    
    def __call__(self, p):
        #------------------------------------------------------------------
        # Options 
        #------------------------------------------------------------------
        opts=Opts(
            dim=p.meta_data.n_variables, 
            Instance=p.meta_data.instance,
            Index=p.meta_data.problem_id,   
            Name=p.meta_data.name, 
            ShowIts=0,
            MaxEvals=min(100*p.meta_data.n_variables,1000)+1, 
            MaxIts=float('inf'), 
            Ftarget=float('-inf')
        )
        opts_dict=opts.to_dict()
        opts_matlab=eng.eval("struct()")  
        for key, value in opts_dict.items():
            opts_matlab[key]=value  
        #------------------------------------------------------------------  
        # Bounds
        #------------------------------------------------------------------
        lower_bounds=p.bounds.lb
        upper_bounds=p.bounds.ub
        bounds=np.array([[lb,ub] for lb, ub in zip(lower_bounds,upper_bounds)])
        #------------------------------------------------------------------
        # Run algorithm
        #------------------------------------------------------------------
        time.sleep(1)
        values_matlab=eng.alg_RS(1, opts_matlab, bounds) 
        f_values_list=np.array(values_matlab)  
        #------------------------------------------------------------------
        # Assigning calculations from MATLAB to IOH
        #------------------------------------------------------------------
        for row in f_values_list:
            global f_value
            f_value=row[0]
            x_value=row[1:]
            p(x_value)

#==========================================================================
Suite=["bench_fun_eps","bench_fun_pitz","Circular_Antenna_Array",
       "Frequency_Modulated_Sound_Waves","Frequency_Modulated_Sound_Waves",
       "Frequency_Modulated_Sound_Waves","Frequency_Modulated_Sound_Waves",
       "Lennard_Jones_Potential","Lennard_Jones_Potential","Lennard_Jones_Potential",
       "Lennard_Jones_Potential","Spacecraft_Trajectory_OptimizationC1",
       "Spacecraft_Trajectory_OptimizationC2","Spread_Spectrum_Radar",
       "Tersoff_PotentialC1","Tersoff_PotentialC1","Tersoff_PotentialC1",
       "Tersoff_PotentialC1","WindWake","WindWake","WindWake","WindWake"]
Ndims=[49,10,12,6,12,24,48,6,12,24,48,26,22,20,6,12,24,48,6,12,24,48]

for fun_name, dim in zip(Suite, Ndims):
    globals()[fun_name]=lambda x: f_value
    ioh.wrap_problem(
        globals()[fun_name],
        name=fun_name,
        optimization_type=ioh.OptimizationType.MIN,
        problem_class=ProblemClass.REAL
    )
    
    def run_experiment():
        exp=Experiment(
            Algorithm(),
            fids=[fun_name],
            dims=[dim],
            iids=[0],
            njobs=1,
            reps=15,
            old_logger=False,
            store_positions=False,
            remove_data=False,
            algorithm_name="RS",
            algorithm_info="v1.0",
            experiment_attributes={
                "test_suite": "Practical problem suite",
                "evaluation_budget": "min(1000, dims[i] * 100)",
            },
        )
    
        # Run the experiment
        exp()
        
    if __name__ == "__main__":
        run_experiment()

if os.path.exists("ioh_data"):
    shutil.rmtree("ioh_data")

eng.quit() # Quit MATLAB after finishing
print("Successful launch")