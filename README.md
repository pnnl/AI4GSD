# AI4GSD
An end-to-end vision AI toolkit for mapping sediment grain-size distributions from drone and smartphone photos and videos.

## Introduction
AI4GSD automates sediment grain-size mapping from drone and smartphone videos and photos. For each image, three artificial intelligence (AI) models—Scale AI, Anthro AI, and Grain AI—detect reference scales, anthropogenic objects, and individual grains, respectively. Scale AI combines detected scales with user-provided dimensions to estimate pixel size (mm/pixel). Anthro AI identifies anthropogenic objects for exclusion from grain analysis, and Grain AI measures individual grains in pixels, which are converted to physical dimensions using the estimated pixel size.

AI4GSD has been evaluated on both drone and smartphone imagery. Across camera types, pixel-size estimates achieved a Nash–Sutcliffe efficiency (NSE) of 0.999, a mean percent error (MPE) of −0.05%, and a mean absolute percent error (MAPE) of 0.65%. For median grain size (D50), the corresponding values were 0.984, 0.77%, and 3.50%. For drone images, D50 estimates achieved an NSE of 0.995, an MPE of 1.26%, and a MAPE of 2.85%; for smartphone images, the corresponding values were 0.979, 0.34%, and 4.09%. Comparisons with a BaseGrain tutorial example and published field observations yielded reported errors of −7.59% and 1.17%, respectively.

Each image typically provides hundreds to thousands of grain measurements. AI4GSD uses these measurements to generate grain-size distributions based on grain count or area weighting and calculate characteristic grain sizes, including D10, D50, D60, and D84. It exports the resulting statistics in standard formats and overlays the predicted reference scales, anthropogenic objects, and grains on the original images for review. With figure generation disabled, average processing time is approximately 2 seconds per frame. The accompanying demos illustrate the tool’s functions in Google Colab, Jupyter Notebook, and Python scripts.

## Funding
This work is supported by PNNL’s River Corridor Hydrobiogeochemistry Science Focus Area project (https://www.pnnl.gov/projects/river-corridor), funded by the U.S. Department of Energy’s Biological and Environmental Research (BER) program.

## References

Details of the models and associated datasets are available in the following publications and data releases:

Chen Y ; Renteria L ; Chen R ; Li B ; Forbes B ; Goldman A E (2025): Videos, photos, and AI-derived grain size data associated with “Modular AI and video surveys transform multiscale sediment grain size mapping”. River Corridor Hydro-biogeochemistry from Molecular to Multi-Basin Scales SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/2588483 

Chen, Y., Bao, J., Chen, R., Li, B., Yang, Y., Renteria, L., et al. (2024). Quantifying streambed grain size, uncertainty, and hydrobiogeochemical parameters using machine learning model YOLO. Water Resources Research, 60, e2023WR036456. https://doi.org/10.1029/2023WR036456

Chen Y ; Bao J ; Chen Y ; Li B ; Yang Y ; Stegen J C ; Renteria L ; Delgado D ; Forbes B ; Goldman A E ; Simhan M ; Barnes M ; Laan M ; McKever S A ; Hou Z ; Chen X ; Scheibe T D (2023): Artificial intelligence models, photos, and data associated with the manuscript “Quantifying Streambed Grain Size, Uncertainty, and Hydrobiogeochemical Parameters Using Machine Learning Model YOLO” (v2). River Corridor and Watershed Biogeochemistry SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/1999774

## Run AI4GSD

### Option 1: Google Colab

Please open [Demo_Colab](Demo_Colab) or select one of the links below to start a demo. Once the notebook opens, click **Copy to Drive** below the **Help** menu to save a copy to your Google Drive. By default, the copy is saved in **My Drive → Colab Notebooks**.

* [Demo 1: Drone Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo1_Drone_Videos.ipynb) — Process drone videos.
* [Demo 2: Smartphone Photos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo2_Smartphone_Photos.ipynb) — Process smartphone photos.
* [Demo 3: Smartphone Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo3_Smartphone_Videos.ipynb) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo4_Smartphone_PhotosAndVideos.ipynb) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo5_FieldValidate_BaseGrain.ipynb) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair2024 Site K1 Validation](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo6_FieldValidate_Mair2024SiteK1.ipynb) — Validate AI predictions against published field measurements from Site K1.
* [Demo 7: Run on Your Own Photos and Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo7_Run_From_Your_Folder.ipynb) — Process your photos and videos containing scale keychains from PNNL. In **My Drive** in Google Drive, create a folder (e.g., `20261113_YC_CSF`) with two subfolders: `photos` and `videos`. Upload your photos and videos to the corresponding subfolders. Then open Demo 7 and replace the folder name in **Step 2** with your actual folder name.

### Option 2: Google Colab for AGU26 Workshop

These demos support Part 1, Module 1: Sediment Grain Size Mapping from Video, in the AGU 2026 workshop [SCIWS02 – AI-Enabled Multiscale Environmental Science: From Smartphone Videos to Satellite-Derived Predictions to Geospatial Foundation Models](https://studio.m-anage.com/agu/agu26/meetingapp.cgi/Session/280975).

Open [Demo_Colab](Demo_Colab) or select one of the links below to start a demo. Once the notebook opens, click **Copy to Drive** below the **Help** menu to save a copy to your Google Drive. By default, the copy is saved in **My Drive → Colab Notebooks**.
* [Demo 1: Drone Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo1_Drone_Videos.ipynb) — Process drone videos.
* [Demo 2: Smartphone Photos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo2_Smartphone_Photos.ipynb) — Process smartphone photos.
* [Demo 3: Smartphone Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo3_Smartphone_Videos.ipynb) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo4_Smartphone_PhotosAndVideos.ipynb) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo5_FieldValidate_BaseGrain.ipynb) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair2024 Site K1 Validation](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo6_FieldValidate_Mair2024SiteK1.ipynb) — Validate AI predictions against published field measurements from Site K1.
* [Demo 7: Run on Your Own Photos and Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo7_Run_From_Your_Folder.ipynb) — Process your photos and videos containing scale keychains from PNNL. In **My Drive** in Google Drive, create a folder (e.g., `20261113_YC_CSF`) with two subfolders: `photos` and `videos`. Upload your photos and videos to the corresponding subfolders. Then open Demo 7 and replace the folder name in **Step 2** with your actual folder name.

### Option 3: Local Jupyter Notebook

For Windows, please follow the steps in [Install_Windows.md](Install_Windows.md) to install AI4GSD before running the demos.
For Linux,   please follow the steps in [Install_Linux.md](Install_Linux.md) to install AI4GSD before running the demos.
For Windows, please follow the steps in [Install_macOS.md](Install_macOS.md) to install AI4GSD before running the demos.

* [Demo 1: Drone Videos](Demo_JupyterNotebook/Demo1_Drone_Videos.ipynb) — Process drone videos.
* [Demo 2: Smartphone Photos](Demo_JupyterNotebook/Demo2_Smartphone_Photos.ipynb) — Process smartphone photos.
* [Demo 3: Smartphone Videos](Demo_JupyterNotebook/Demo3_Smartphone_Videos.ipynb) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](Demo_JupyterNotebook/Demo4_Smartphone_PhotosAndVideos.ipynb) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](Demo_JupyterNotebook/Demo5_FieldValidate_BaseGrain.ipynb) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair2024 Site K1 Validation](Demo_JupyterNotebook/Demo6_FieldValidate_Mair2024SiteK1.ipynb) — Validate AI predictions against published field measurements from Site K1.

### Option 4: Python scripts in Spyder

For Windows, please follow the steps in [Install_Windows.md](Install_Windows.md) to install AI4GSD before running the demos.
For Linux,   please follow the steps in [Install_Linux.md](Install_Linux.md) to install AI4GSD before running the demos.
For Windows, please follow the steps in [Install_macOS.md](Install_macOS.md) to install AI4GSD before running the demos.

* [Demo 1: Drone Videos](Demo_Python_Release/Demo1_Drone_Videos.py) — Process drone videos.
* [Demo 2: Smartphone Photos](Demo_Python_Release/Demo2_Smartphone_Photos.py) — Process smartphone photos.
* [Demo 3: Smartphone Videos](Demo_Python_Release/Demo3_Smartphone_Videos.py) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](Demo_Python_Release/Demo4_Smartphone_PhotosAndVideos.py) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](Demo_Python_Release/Demo5_FieldValidate_BaseGrain.py) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair2024 Site K1 Validation](Demo_Python_Release/Demo6_FieldValidate_Mair2024SiteK1.py) — Validate AI predictions against published field measurements from Site K1.

## Citation

```bibtex
@misc{Chen2025,
address = {Richland, WA, USA},
author = {Chen, Yunxiang},
title = {{AI4GSD: an end-to-end vision AI toolkit to map sediment grain size distribution from drone and smartphone photos and videos}},
url = {https://github.com/pnnl/AI4GSD },
year = {2025}
}
```

## Developer
Dr. Yunxiang Chen (yunxiang.chen@pnnl.gov), Pacific Northwest National Laboratory, Richland, WA, USA 

## License
The project uses the following licenses:

* All `*.py` files in `./AI4GSD`: Apache 2.0; see `./Licenses`.
* All `*.csv` files in `./AIGSD/templates`: Apache 2.0; see `./Licenses`.
* All `*.pt` files in `./AIGSD/templates`: Apache 2.0; see `./Licenses`.
* All `*.csv` files in `./Scales`: Apache 2.0; see `./Licenses`.
* All files in `./Demo_Data` and `./Demo_JypyterNotebook`: BSD; see `./Licenses`.
* The bundled software in `ffmpeg`, `exiftool_linux`, and `exiftool_windows` under `./AI4GSD/exes` is distributed under its respective licenses.
