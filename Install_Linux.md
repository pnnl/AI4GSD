# AI4GSD Installation Guide for Ubuntu 20.04

Install AI4GSD, then run the demos using **Jupyter Notebook** or **Spyder**.

This guide covers **64-bit Ubuntu Linux on Intel/AMD processors**, using **Ubuntu 20.04** as an example. Run all commands in **Terminal**.

Choose CPU processing or an NVIDIA GPU installation with a compatible driver.

> **Validation:** These instructions adapt the Windows guide using documented Linux installation options. The complete AI4GSD workflow has not been tested on Ubuntu 20.04 as part of this guide.

## Step 0 · Install Anaconda

Skip this step if Anaconda is already installed and `conda --version` works.

1. Download the **Linux x86_64** shell installer from the [Anaconda download page](https://www.anaconda.com/download/success).
2. In Terminal, navigate to the download folder and run the installer. For example, if you downloaded this version:

```bash
cd ~/Downloads
bash Anaconda3-2026.07-1-Linux-x86_64.sh
```

3. Follow the prompts. Accept shell initialization when prompted.
4. Close and reopen Terminal.

Replace the installer filename with the exact filename you downloaded. Follow the installer’s listed CPU and operating-system requirements.

## Step 1 · Open Terminal and check Conda

Open **Terminal**, or press **Ctrl+Alt+T**.

Check Conda:

```bash
conda --version
```

If Conda is installed but not recognized, initialize your shell using the full path to your Conda executable. Examples:

```bash
# Ubuntu: common installation path; default shell is usually bash.
~/anaconda3/bin/conda init bash
```

Adjust the installation path if needed, then reopen Terminal.

## Step 2 · Create the AI4GSD environment

```bash
conda create -n ai4gsd python=3.11.13 -y
```

## Step 3 · Activate the environment

```bash
conda activate ai4gsd
```

The prompt should now begin with `(ai4gsd)`.

> Activate this environment whenever you open a new prompt to use AI4GSD.

## Step 4 · Install the core packages

```bash
python -m pip install numpy==2.2.6 pandas==2.3.1 matplotlib==3.10.6
```

## Step 5 · Install additional packages and external tools

Install the Python packages:

```bash
python -m pip install natsort pyproj PyExifTool piexif imutils shapely psutil gdown
```

**PyExifTool is a Python wrapper; the ExifTool executable must also be available.** Installing FFmpeg and ExifTool separately provides system executables for video processing and metadata extraction.

```bash
sudo apt update
sudo apt install -y ffmpeg libimage-exiftool-perl libgl1 libglib2.0-0
```

Check the executables:

```bash
ffmpeg -version
exiftool -ver
```

## Step 6 · Install PyTorch

Choose **one** option based on your hardware. If you are unsure, use **Option A: CPU**.

### Option A: CPU

Use this option if you do not have a compatible NVIDIA GPU, or if you prefer CPU processing.

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

### Option B: NVIDIA GPU — CUDA 12.6

Check that the NVIDIA driver recognizes your GPU:

```bash
nvidia-smi
```

With a compatible GPU and driver, install:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

### Option C: NVIDIA GPU — CUDA 13.0

Requires a GPU and driver supported by the CUDA 13.0 build:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

A newer CUDA build is not necessarily compatible with an older GPU. Check the [PyTorch installation options](https://pytorch.org/get-started/previous-versions/) and [NVIDIA CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/) before choosing a GPU build. The prebuilt PyTorch packages include their CUDA runtime dependencies; ordinary use does not require installing a separate CUDA toolkit.

## Step 7 · Install OpenCV

```bash
python -m pip install opencv-python==4.12.0.88
```

## Step 8 · Install Ultralytics

```bash
python -m pip install ultralytics==8.3.186
```

## Step 9 · Install Jupyter Notebook

Install Jupyter Notebook and register the AI4GSD environment as a notebook kernel.

```bash
python -m pip install notebook ipykernel
python -m ipykernel install --user --name ai4gsd --display-name "Python (ai4gsd)"
```

## Step 10 · Verify the installation

Check for dependency conflicts:

```bash
python -m pip check
```

If no conflicts are detected, the output should be:

```text
No broken requirements found.
```

Print the installed package versions and check GPU availability:

```bash
python - <<'PYTHON'
import platform
import sys
import numpy
import pandas
import cv2
import matplotlib
import ultralytics
import torch
import torchvision

print("Python:", sys.version.split()[0])
print("Architecture:", platform.machine())
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("opencv:", cv2.__version__)
print("matplotlib:", matplotlib.__version__)
print("ultralytics:", ultralytics.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
PYTHON
```

| Installation | Expected availability |
|---|---|
| Linux, CPU build | CUDA: `False` |
| Linux, working NVIDIA GPU installation | CUDA: `True` |

These checks verify package imports and backend availability. Run a demo to check AI4GSD itself.

## Step 11 · Download AI4GSD

1. Open the [AI4GSD GitHub repository](https://github.com/pnnl/AI4GSD).
2. Click the green **Code** button and select **Download ZIP**.
3. Extract the ZIP file.
4. Rename the extracted folder to **AI4GSD** and move it to your preferred location, such as `~/AI4GSD`.

Your demo folders will then be:

| Interface | Example folder |
|---|---|
| Jupyter Notebook | `~/AI4GSD/Demo_JupyterNotebook` |
| Spyder | `~/AI4GSD/Demo_Python_Release` |

## Step 12 · Run the Jupyter Notebook demos

In Terminal, activate the environment and navigate to the notebook folder:

```bash
conda activate ai4gsd
cd ~/AI4GSD/Demo_JupyterNotebook
jupyter notebook
```

Replace `~/AI4GSD` with your actual installation path.

When Jupyter opens in your browser:

1. Open a demo notebook (`.ipynb`).
2. Select **Python (ai4gsd)** as the kernel.
3. Follow the notebook’s instructions and run its cells in order.

## Step 13 · Run the Python demos in Spyder

This step is optional if you only use Jupyter Notebook.

### Open or install Spyder

If Spyder is already installed, open it from your application launcher.

Otherwise, install it in a separate Conda environment so its dependencies are managed independently of AI4GSD:

```bash
conda create -n spyder-env -c conda-forge python=3.11 spyder -y
conda activate spyder-env
spyder
```

For subsequent launches:

```bash
conda activate spyder-env
spyder
```

If Conda cannot resolve a Spyder build for your operating system, check the [Spyder installation guide](https://docs.spyder-ide.org/current/installation.html) for supported versions. Current Spyder recommendations list Ubuntu 22.04; Ubuntu 20.04 may require an older compatible Spyder release. Jupyter Notebook remains an alternative interface.

### Select the AI4GSD interpreter

First, find the environment’s Python executable by running:

```bash
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
2. Open a `.py` script from `~/AI4GSD/Demo_Python_Release`.
3. Follow the script’s instructions and set any required paths.
4. Press **F5** to run it.

To confirm that Spyder is using the correct environment, run this in its IPython console:

```python
import sys
print(sys.executable)
```

The path should end with `envs/ai4gsd/bin/python`.

> When returning to Jupyter Notebook or installing AI4GSD packages, run `conda activate ai4gsd` first.


## Platform notes

- Use forward slashes in paths, such as `/home/yourname/AI4GSD`.
- Folder and filename capitalization must match exactly on case-sensitive filesystems, including typical Ubuntu installations.
- Choose an AI4GSD folder where your account can write output files.
- If a demo contains Windows-specific paths or hard-coded executable locations, update them for your system. Successful dependency installation alone does not establish that those code paths support your operating system.

## Installation references

- [Conda installation on Linux](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
- [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)
- [Spyder installation guide](https://docs.spyder-ide.org/current/installation.html)
- [Spyder environment instructions](https://docs.spyder-ide.org/current/faq.html?highlight=interpreter)
