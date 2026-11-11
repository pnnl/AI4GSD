# AI4GSD Installation Guide for Windows

Follow these steps to install AI4GSD and launch its Jupyter Notebook demos.

## Step 0: Install Anaconda

Skip this step if Anaconda is already installed.

1. Download the [Anaconda installer for Windows](https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Windows-x86_64.exe).
2. Double-click the downloaded `.exe` file and follow the installation instructions.

## Step 1: Open Anaconda PowerShell Prompt

Search for **Anaconda PowerShell Prompt** in the Windows Start menu and open it.

Run all commands below in this prompt.

## Step 2: Create an AI4GSD environment

```powershell
conda create -n ai4gsd python=3.11.13 -y
```

## Step 3: Activate the environment

```powershell
conda activate ai4gsd
```

Activate this environment whenever you open a new prompt to use AI4GSD.

## Step 4: Install packages with specified versions

```powershell
python -m pip install numpy==2.2.6 pandas==2.3.1 matplotlib==3.10.6
```

## Step 5: Install additional packages

```powershell
python -m pip install natsort pyproj PyExifTool piexif imutils shapely psutil gdown
```

## Step 6: Install PyTorch

Choose **one** option based on your hardware. If you are unsure, use **Option A: CPU**.

### Option A: CPU

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

### Option B: NVIDIA GPU — CUDA 12.6 build

Requires a compatible NVIDIA GPU and driver.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

### Option C: NVIDIA GPU — CUDA 13.0 build

Requires a compatible NVIDIA GPU and driver.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

## Step 7: Install OpenCV

```powershell
python -m pip install opencv-python==4.12.0.88
```

## Step 8: Install Ultralytics

```powershell
python -m pip install ultralytics==8.3.186
```

## Step 9: Install Jupyter Notebook

Install Jupyter Notebook and register the AI4GSD environment as a notebook kernel.

```powershell
python -m pip install notebook ipykernel
python -m ipykernel install --user --name ai4gsd --display-name "Python (ai4gsd)"
```

## Step 10: Verify the installation

Check for dependency conflicts:

```powershell
python -m pip check
```

Print the installed package versions and GPU availability:

```powershell
python -c "import numpy, pandas, cv2, matplotlib, ultralytics, torch, torchvision; print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('opencv:', cv2.__version__); print('matplotlib:', matplotlib.__version__); print('ultralytics:', ultralytics.__version__); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For a CPU installation, `CUDA available: False` is expected.

## Step 11: Download the code

Navigate to the green `Code` button, click it, and select "Download ZIP". Extracting the file and rename it to "AI4GSD".

## Step 12: Launch Jupyter Notebook

Navigate to the `Demo_JupyterNotebook` folder inside your AI4GSD directory (For example: D:\). For example:

```powershell
cd D:\AI4GSD\Demo_JupyterNotebook
jupyter notebook
```

In the browser, open a demo notebook and select **Python (ai4gsd)** as its kernel.

## Step 13: Launch Spyder

Find "Spyder" from Windows search or Start, and open it to launch. Then Select Open File in Spyder to open .py file from Demo_Python_Release. Please change the Python interprestr to C:/Users/[YOUR COMPUTER NAME]/AppData/Local/anaconda3/envs/ai4gsd/python.exe

Navigate to the `Demo_JupyterNotebook` folder inside your AI4GSD directory (For example: D:\). For example:

```powershell
cd D:\AI4GSD\Demo_JupyterNotebook
jupyter notebook
```
In the lower right corner of Spyder, you should see **conda (ai4gsd)** as its interpreter.

You are now ready to run the AI4GSD demos.
