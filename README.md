# AI4GSD
An end-to-end vision AI toolkit to map sediment grain size distribution from drone and smartphone photos and videos.

## Introduction
AI4GSD is a computer-vision tool for automated grain-size mapping from drone- and smartphone-acquired videos and photographs. For each image, it uses three artificial intelligence (AI) models—Scale AI, Anthro AI, and Grain AI—to detect reference scales, anthropogenic objects, and individual grains, respectively. User-provided reference-scale dimensions are used to convert the detected scale pixels to pixel size (mm/pixel). Anthropogenic-object regions are excluded from the grain analysis, and the pixel sizes of individual grains are converted to physical sizes using the pixel size.

AI4GSD has been evaluated on both drone and smartphone imagery. Across camera types, pixel-size estimates achieved a Nash–Sutcliffe efficiency (NSE) of 0.999, a mean percent error (MPE) of −0.05%, and a mean absolute percent error (MAPE) of 0.65%. For median grain size, the corresponding values were 0.984, 0.77%, and 3.50%. For drone images, median grain-size estimates achieved an NSE of 0.995, an MPE of 1.26%, and a MAPE of 2.85%; for smartphone images, the corresponding values were 0.979, 0.34%, and 4.09%. The tool was also tested against a BaseGrain tutorial example and field-observation data reported in the literature, with reported errors of −7.59% and 1.17%, respectively.

Each image typically yields hundreds to thousands of grain measurements. These measurements are used to generate grain-size distributions by grain count or area weighting, from which characteristic grain sizes are calculated, including D10, D50, D60, and D84. AI4GSD exports the resulting statistics in standard formats and provides an overlay of the predicted reference scales, anthropogenic objects, and individual grains on each original image for user review. With figure generation disabled, the average processing time is approximately 2 seconds per frame. Users can explore the tool’s functions through the accompanying Jupyter Notebook examples.

## Funding
The work is supported by PNNL’s River Corridor Hydrobiogeochemistry Science Focus Area project (https://www.pnnl.gov/projects/river-corridor) funded by US DOE Biological and Environmental Research (BER)

## Reference
Details of the data and model can be found in:

Chen Y ; Renteria L ; Chen R ; Li B ; Forbes B ; Goldman A E (2025): Videos, photos, and AI-derived grain size data associated with “Modular AI and video surveys transform multiscale sediment grain size mapping”. River Corridor Hydro-biogeochemistry from Molecular to Multi-Basin Scales SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/2588483 

Chen, Y., Bao, J., Chen, R., Li, B., Yang, Y., Renteria, L., et al. (2024). Quantifying streambed grain size, uncertainty, and hydrobiogeochemical parameters using machine learning model YOLO. Water Resources Research, 60, e2023WR036456. https://doi.org/10.1029/2023WR036456

Chen Y ; Bao J ; Chen Y ; Li B ; Yang Y ; Stegen J C ; Renteria L ; Delgado D ; Forbes B ; Goldman A E ; Simhan M ; Barnes M ; Laan M ; McKever S A ; Hou Z ; Chen X ; Scheibe T D (2023): Artificial intelligence models, photos, and data associated with the manuscript “Quantifying Streambed Grain Size, Uncertainty, and Hydrobiogeochemical Parameters Using Machine Learning Model YOLO” (v2). River Corridor and Watershed Biogeochemistry SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/1999774

## Run the Code

### Option 1: Jupyter Notebook

Follow the steps in [Install.md](Install.md) to install AI4GSD before running the demos.

Open [Demo_JupyterNotebook](Demo_JupyterNotebook), select a notebook, and follow its instructions. Example data are provided in [Demo_Data](Demo_Data).

* [Demo 1: Drone Videos](Demo_JupyterNotebook/Demo1_Drone_Videos.ipynb) — Process drone videos.
* [Demo 2: Smartphone Photos](Demo_JupyterNotebook/Demo2_Smartphone_Photos.ipynb) — Process smartphone photos.
* [Demo 3: Smartphone Videos](Demo_JupyterNotebook/Demo3_Smartphone_Videos.ipynb) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](Demo_JupyterNotebook/Demo4_Smartphone_PhotosAndVideos.ipynb) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](Demo_JupyterNotebook/Demo5_FieldValidation_BaseGrainExample.ipynb) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair (2024) Site K1 Validation](Demo_JupyterNotebook/Demo6_FieldValidation_Mair2024SiteK1.ipynb) — Validate AI predictions against published field measurements from Site K1.

### Option 2: Google Colab

Open a demo using the links below and run its cells in order. Each notebook includes instructions for installing packages and downloading the code and required data.

* [Demo 1: Drone Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo1_Drone_Videos.ipynb) — Process drone videos.
* [Demo 2: Smartphone Photos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo2_Smartphone_Photos.ipynb) — Process smartphone photos.
* [Demo 3: Smartphone Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo3_Smartphone_Videos.ipynb) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo4_Smartphone_PhotosAndVideos.ipynb) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo5_FieldValidation_BaseGrainExample.ipynb) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair (2024) Site K1 Validation](https://colab.research.google.com/github/pnnl/AI4GSD/blob/main/Demo_Colab/Demo6_FieldValidation_Mair2024SiteK1.ipynb) — Validate AI predictions against published field measurements from Site K1.

### Option 3: Spyder

Follow the steps in [Install.md](Install.md) to install AI4GSD before running the demos.

Export one of the notebooks below as a Python script (`.py`), then open it in Spyder. Run any package installation commands in your Anaconda prompt. Example data are provided in [Demo_Data](Demo_Data).

* [Demo 1: Drone Videos](Demo_JupyterNotebook/Demo1_Drone_Videos.ipynb) — Process drone videos.
* [Demo 2: Smartphone Photos](Demo_JupyterNotebook/Demo2_Smartphone_Photos.ipynb) — Process smartphone photos.
* [Demo 3: Smartphone Videos](Demo_JupyterNotebook/Demo3_Smartphone_Videos.ipynb) — Process smartphone videos.
* [Demo 4: Smartphone Photos and Videos](Demo_JupyterNotebook/Demo4_Smartphone_PhotosAndVideos.ipynb) — Process both smartphone photos and videos.
* [Demo 5: BaseGrain Validation](Demo_JupyterNotebook/Demo5_FieldValidation_BaseGrainExample.ipynb) — Validate AI predictions against the BaseGrain tutorial example.
* [Demo 6: Mair (2024) Site K1 Validation](Demo_JupyterNotebook/Demo6_FieldValidation_Mair2024SiteK1.ipynb) — Validate AI predictions against published field measurements from Site K1.

## Cite as

@misc{Chen2025,
address = {Richland, WA, USA},
author = {Chen, Yunxiang},
title = {{AI4GSD: an end-to-end vision AI toolkit to map sediment grain size distribution from drone and smartphone photos and videos}},
url = {https://github.com/pnnl/AI4GSD },
year = {2025}
}

## Developers
Dr. Yunxiang Chen (yunxiang.chen@pnnl.gov), Pacific Northwest National Laboratory, Richland, WA, USA 

## License
This code applies dual licensing:

* All *.py files within in ./AI4GSD folder:  Apache 2.0 license, see details in ./Licenses
* All *.csv files within in ./AIGSD/templates: Apache 2.0 license, see details in ./Licenses
* All *.pt files within in ./AIGSD/templates: Apache 2.0 license, see details in ./Licenses
* All *.csv files within in ./Scales: Apache 2.0 license, see details in ./Licenses
* All files in ./Demo_Data and ./Demo_JypyterNotebook: BSD license, see details in ./Licenses
* Files in ffmpeg, exiftool_linux, exiftool_windows in ./AI4GSD/exes are open-source software, following their own license.