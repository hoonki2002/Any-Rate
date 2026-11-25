"""Run a script of the authors' code unchanged: python launch.py [--tf32] [--stand-in MODULE] SCRIPT ARGS...  (from the folder of that code)
  --tf32              TF32 matmuls, as in our AnyBCQ runs (the released script leaves the default)
  --stand-in MODULE   an empty stand-in for a module that the script imports but that our runs never use, if it is not installed:
                      custom_kernel (AnyBCQ inference kernels, a CUDA build) and wandb (MatGPTQ logging, off unless --log_wandb)"""
import importlib.machinery, importlib.util, os, runpy, sys, types


def stand_in(name):
    if importlib.util.find_spec(name) is not None: return
    m = types.ModuleType(name); m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    def missing(attr):
        if attr.startswith("__"): raise AttributeError(attr)
        raise RuntimeError(f"{name}.{attr} was called, but only a stand-in of {name} is installed")
    m.__getattr__ = missing; sys.modules[name] = m


args = sys.argv[1:]
while args[0].startswith("--"):
    if args[0] == "--tf32":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high"); args = args[1:]
    elif args[0] == "--stand-in":
        stand_in(args[1]); args = args[2:]
    else:
        sys.exit(f"unknown option {args[0]}")
sys.argv = args; sys.path.insert(0, os.getcwd())
runpy.run_path(args[0], run_name="__main__")
