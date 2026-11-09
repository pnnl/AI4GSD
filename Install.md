# Step 0: install Anaconda if you have not installed it.
Dowmload Anaconda from https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Windows-x86_64.exe
Double click the downloaded .exe file to install it

# Step 1: open Anaconda powershell prompt
Type anaconda Powershell Prompt in Windoers search, and click it.

# Step 2: create a new environment for AI4GSD.
conda create -n ai4gsd python=3.11.13 -y

# Step 3: activate the environment.
conda activate ai4gsd

# Step 4: install packages with version requirements.
python -m pip install numpy==2.2.6 pandas==2.3.1 matplotlib==3.10.6

# Step 5: install packages without version requirements.
python -m pip install natsort pyproj PyExifTool piexif imutils shapely psutil

# Step 6: install PyTorch based on hardward.
## Option A: run on CPU
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu

## Option B: run on NVIDIA GPU with CUDA 12.1
# python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

## Option C: run on NVIDIA GPU with CUDA 11.8
# python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu118

# Step 7: install opencv
python -m pip install opencv-python==4.12.0.88

# Step 8: install Ultralytics
python -m pip install ultralytics==8.3.186

# Step 9: install Jupyter Notebook.
python -m pip install notebook
python -m ipykernel install --user --name ai4gsd --display-name "Python (ai4gsd)"

# Step 10: verify installation (optional).
python -c "import numpy,pandas,cv2,matplotlib,ultralytics,torch,torchvision; \
print('numpy',numpy.__version__); \
print('pandas',pandas.__version__); \
print('cv2',cv2.__version__); \
print('matplotlib',matplotlib.__version__); \
print('ultralytics',ultralytics.__version__); \
print('torch',torch.__version__); \
print('torchvision',torchvision.__version__); \
print('cuda available:', torch.cuda.is_available())"

# Step 11: launch Jupyter Notebook.
cd to Demo_JupyterNotebook folder, for example: cd D:\AI4GSD\Demo_JupyterNotebook
jupyter notebook

# End.