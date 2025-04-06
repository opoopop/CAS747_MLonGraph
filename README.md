# Readme

This is the code of LLM-GNN. The code is build from scratch.

./notebooks include some jupyter notebooks with the result.  notebooks/citeseer_prompt_test.ipynb & notebooks/Cora_prompt_test.ipynb is result of section C  and notebooks/Cora_post_filter.ipynb is the section D.

Steps to run the code and quickly test the result.

1. create virtual environment and install the requirements

   ```bash
   conda create -n myenv python=3.9
   conda activate myenv
   ```

   ```bash
   cd path/to/your/file
   pip install -r requirements.txt
   ```

   for pytorch it need torch==2.5.0, find your version from [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

   ```python
   # ROCM 6.1 (Linux only)
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.1
   # ROCM 6.2 (Linux only)
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2
   # CUDA 11.8
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
   # CUDA 12.4
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
   # CPU only
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
   ```

   then get torch_geometric. find your version from [Installation — pytorch_geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

   ```python
   pip install torch_geometric
   
   # Optional dependencies:
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
   ```

   then 

   ```python
   conda install mkl mkl-service
   ```

   finally the get the right faiss

   ```python
   conda install faiss-gpu
   #conda install faiss-cpu # this is cpu version
   ```

   

2. main.py is the code to test Cora. 

   Example using few shot prompt:

   ```python
   python3 src/main.py
   ```

   result:

   ```
   suucessfully read the data
   accuracy of all the annotations: 70.98 %
   finish preparing the data and response from LLM using few_shot prompt, 541 nodes in total
   start train GNN
   Epoch 001, Loss: 2.3132, Train: 0.1275, Val: 0.1520
   Epoch 020, Loss: 0.8244, Train: 0.7726, Val: 0.6760
   Epoch 040, Loss: 0.6407, Train: 0.8115, Val: 0.7020
   Epoch 060, Loss: 0.5598, Train: 0.8447, Val: 0.7220
   final accuracy of GNN: 72.2 %
   ```

   if you want to use post filtering ,and wait for 1 mins

   ```python
   python3 src/main.py --post_filter True
   ```

   result:

   ```python
   suucessfully read the data
   accuracy of all the annotations: 70.98 %
   finish preparing the data and response from LLM using few_shot prompt, 541 nodes in total
   start post filtering process
   process 40 nodes
   process 80 nodes
   process 120 nodes
   process 160 nodes
   process 200 nodes
   process 240 nodes
   300 nodes left
   start train GNN
   Epoch 001, Loss: 2.3452, Train: 0.1433, Val: 0.1420
   Epoch 020, Loss: 0.6369, Train: 0.8367, Val: 0.6840
   Epoch 040, Loss: 0.4575, Train: 0.8800, Val: 0.6880
   Epoch 060, Loss: 0.3784, Train: 0.9067, Val: 0.7140
   final accuracy of GNN: 74.2 %
   
   ```

   Here are some paprmeters you can chose

   ```python
   --post_filter #True or False , use post filtering or not
   --prompt # few_shot or zero_shot or topk or consistency
   --filter_num # number of nodes left after post filtering , default=300
   ```

   The test result is on result/result.txt
