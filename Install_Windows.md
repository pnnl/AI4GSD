# AI4GSD Installation Guide for Windows

Install AI4GSD, then run the demos using **Jupyter Notebook** or **Spyder**.

> **Before you begin:** Run all installation commands in **Anaconda PowerShell Prompt**. Install only **one** PyTorch option in Step 6.

## Step 0 · Install Anaconda

Skip this step if Anaconda is already installed.

1. Download the [Anaconda installer for Windows](https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Windows-x86_64.exe).
2. Double-click the downloaded `.exe` file and follow the installation instructions.

## Step 1 · Open Anaconda PowerShell Prompt

Search for **Anaconda PowerShell Prompt** in the Windows Start menu and open it.

Run the commands below in this prompt.

## Step 2 · Create the AI4GSD environment

```powershell
conda create -n ai4gsd python=3.11.13 -y
```

## Step 3 · Activate the environment

```powershell
conda activate ai4gsd
```

The prompt should now begin with `(ai4gsd)`.

> Activate this environment whenever you open a new prompt to use AI4GSD.

## Step 4 · Install the core packages

```powershell
python -m pip install numpy==2.2.6 pandas==2.3.1 matplotlib==3.10.6
```

## Step 5 · Install additional packages

```powershell
python -m pip install natsort pyproj PyExifTool piexif imutils shapely psutil gdown
```

## Step 6 · Install PyTorch

Choose **one** option based on your hardware. If you are unsure, select **Option A: CPU**. These commands follow [PyTorch’s published installation options](https://pytorch.org/get-started/previous-versions/).

### Option A: CPU

Use this option if your computer does not have a compatible NVIDIA GPU.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

### Option B: NVIDIA GPU — CUDA 12.6

Requires a compatible NVIDIA GPU and driver.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

### Option C: NVIDIA GPU — CUDA 13.0

Requires a compatible NVIDIA GPU and driver.

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

## Step 7 · Install OpenCV

```powershell
python -m pip install opencv-python==4.12.0.88
```

## Step 8 · Install Ultralytics

```powershell
python -m pip install ultralytics==8.3.186
```

## Step 9 · Install Jupyter Notebook

Install Jupyter Notebook and register the AI4GSD environment as a notebook kernel.

```powershell
python -m pip install notebook ipykernel
python -m ipykernel install --user --name ai4gsd --display-name "Python (ai4gsd)"
```

## Step 10 · Verify the installation

Check for dependency conflicts:

```powershell
python -m pip check
```

If no conflicts are detected, the output should be:

```text
No broken requirements found.
```

Print the installed package versions and check GPU availability:

```powershell
python -c "import numpy, pandas, cv2, matplotlib, ultralytics, torch, torchvision; print('numpy:', numpy.__version__); print('pandas:', pandas.__version__); print('opencv:', cv2.__version__); print('matplotlib:', matplotlib.__version__); print('ultralytics:', ultralytics.__version__); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

> For a CPU installation, `CUDA available: False` is expected.

## Step 11 · Download AI4GSD

1. Open the [AI4GSD GitHub repository](https://github.com/pnnl/AI4GSD).
2. Click the green **Code** button and select **Download ZIP**.
3. Extract the ZIP file.
4. Rename the extracted folder to **AI4GSD** and move it to your preferred location, such as `D:\AI4GSD`.

Your demo folders will then be:

| Interface | Example folder |
|---|---|
| Jupyter Notebook | `D:\AI4GSD\Demo_JupyterNotebook` |
| Spyder | `D:\AI4GSD\Demo_Python_Release` |

## Step 12 · Run the Jupyter Notebook demos

In Anaconda PowerShell Prompt, activate the environment and navigate to the notebook folder:

```powershell
conda activate ai4gsd
cd "D:\AI4GSD\Demo_JupyterNotebook"
jupyter notebook
```

Replace `D:\AI4GSD` with your actual installation path.

When Jupyter opens in your browser:

1. Open a demo notebook (`.ipynb`).
2. Select **Python (ai4gsd)** as the kernel.
3. Follow the notebook’s instructions and run its cells in order.

## Step 13 · Run the Python demos in Spyder

This step is optional if you only use Jupyter Notebook.

### Open Spyder

Search for **Spyder** in the Windows Start menu and open it. If Spyder is not installed, download its [Windows installer](https://www.spyder-ide.org/). A standalone Spyder installation can connect to your existing `ai4gsd` environment; see the [Spyder installation guide](https://docs.spyder-ide.org/current/installation.html).

### Select the AI4GSD interpreter

First, find the environment’s Python executable by running:

```powershell
conda activate ai4gsd
python -c "import sys; print(sys.executable)"
```

Copy the printed path. In Spyder:

1. Open **Tools → Preferences → Python interpreter**.
2. Select **Use the following Python interpreter**.
3. Paste or browse to the executable path you copied.
4. Click **Apply**, then **OK**.
5. Restart the IPython console or Spyder.

If Spyder reports that `spyder-kernels` is missing or incompatible, run the installation command shown in its message **inside the activated `ai4gsd` environment**, then restart the console. Spyder requires a compatible kernel package to use an external environment; see the [Spyder environment instructions](https://docs.spyder-ide.org/current/faq.html?highlight=interpreter).

### Open and run a demo

1. Select **File → Open**.
2. Open a `.py` script from `D:\AI4GSD\Demo_Python_Release`.
3. Follow the script’s instructions and set any required paths.
4. Press **F5** to run it.

To confirm that Spyder is using the correct environment, run this in its IPython console:

```python
import sys
print(sys.executable)
```

The path should point to `envs\ai4gsd\python.exe`.
