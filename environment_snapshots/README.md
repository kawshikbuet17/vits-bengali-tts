# Environment Snapshots

This folder stores exported environment details from the working remote server setup used for Bengali VITS training.

These files are for reproducibility and debugging only. They are not training data, model checkpoints, or required runtime inputs.

Typical files:

- `python_version.txt`: Python version used in the working environment.
- `torch_cuda_version.txt`: PyTorch version and CUDA runtime version.
- `requirements_server_working.txt`: exact `pip freeze` output from the working environment.
- `conda_list_server_working.txt`: full `conda list` output from the working environment.
- `nvidia_smi_server_working.txt`: GPU, driver, and CUDA driver information.

Known working setup:

- conda environment: `vits-bn310`
- Python: `3.10`
- PyTorch: `2.11.0+cu128`
- training GPU selection: `CUDA_VISIBLE_DEVICES=1`

Keep this folder as a record of the server environment that successfully started Bengali VITS training.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  
