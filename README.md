# AI4GSD
An easy-to-use grain size distribution measurement tool empowered by AI and videos.

## Instructions
AI4GSD is an computer vision AI powered tool for automated grain size distribution measurements using photos and videos from drone and smartphones. This tool has been proven to provide highly accurate results for estimating photo resolution (NSE=1.00, mean percent error (MPE) = 0.14%, mean absolute percent error (MAPE) = 0.95%) and median grain size (NSE=0.98, MPE= -0.03%, MAPE=5.19%) for both drone and smartphone images. For each input image, this tool uses three AI models, i.e., ScaleAI, AnthroAI, and GrainAI, to detect pixel sizes of reference scales, anthropogenic objects, and individual grains, respectively. The reference scale pixel sizes are then converted to photo resolution (mm/pixel) based on user-defined reference scale sizes. The region occupied by the anthropogenic objects is cropped out as non-grain region. The pixel sizes of individual grains are converted to real size by multiplying their pixel sizes with the photo resolution. Each photo typically detects a larger number of grains. Such data are used to generate a grain size distribution based on count or area weight of each grain. Characteristic grain sizes such as D10, D50, D60, D84 are then computed from the grain size distribution curve. Such statistics data are output to standard formats for all photos. The tool also outputs the overlay of AI-predicted reference scale, anthropogenic objects, and individual grains on the raw image for each photo for user's further analyses. 

## Funding
The work is supported by the Program Development Funding from the Physical and Computational Science Directorate (PCSD) of Pacific Northwest National Laboratory (PNNL) and PNNL’s River Corridor Hydrobiogeochemistry Science Focus Area project (https://www.pnnl.gov/projects/river-corridor) funded by US DOE Biological and Environmental Research (BER)

## Reference
Details of the data and model can be found in:

Chen, Y., Bao, J., Chen, R., Li, B., Yang, Y., Renteria, L., et al. (2024). Quantifying streambed grain size, uncertainty, and hydrobiogeochemical parameters using machine learning model YOLO. Water Resources Research, 60, e2023WR036456. https://doi.org/10.1029/2023WR036456

Chen Y ; Renteria L ; Chen R ; Li B ; Forbes B ; Goldman A E (2025): Videos, photos, and AI-derived grain size data associated with “Modular AI and video surveys transform multiscale sediment grain size mapping”. River Corridor Hydro-biogeochemistry from Molecular to Multi-Basin Scales SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/2588483 

Chen Y ; Bao J ; Chen Y ; Li B ; Yang Y ; Stegen J C ; Renteria L ; Delgado D ; Forbes B ; Goldman A E ; Simhan M ; Barnes M ; Laan M ; McKever S A ; Hou Z ; Chen X ; Scheibe T D (2023): Artificial intelligence models, photos, and data associated with the manuscript “Quantifying Streambed Grain Size, Uncertainty, and Hydrobiogeochemical Parameters Using Machine Learning Model YOLO” (v2). River Corridor and Watershed Biogeochemistry SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/1999774

## Installation
The following libraries are used, other version of libraries may also work.  

* Python (3.11.13)
* numpy (2.2.6)
* pandas (2.3.1)
* ultralytics (8.3.186)

Using different version of numpy and pandas as listed above may cause error.

## Setup system path
For Windows and Spyder: open Spyder, then go to "Tools>PYTHONPATH manager>add path". Click the "add path (+ marker)", then add the path of the main folder to the system. The main path is a folder which includes "README.MD".

## Run the code
All the codes for the models are in folder AI4GSD. Four Jupyter Notebooks (in Demo_JupyterNotebook) are provided to demonstrate how to use the code. All the necessary data for the demos are in the folder Demo_Data.

* Demo1_Drone_Videos.ipynb : a demo to show how to process video data from drone.
* Demo2_Smartphone_Photos.ipynb : a demo to show how to process photos from smartphones.
* Demo3_Smartphone_Videos.ipynb : a demo to show how to process video data from smartphones.
* Demo4_Smartphone_PhotosAndVideos.ipynb : a demo to show how to process both photos and videos from smartphone.
  
The users can follow the description in each Jupyter Notebook to run the model step by step.

## Cite as

@misc{Chen2025,
address = {Richland, WA, USA},
author = {Chen, Yunxiang},
title = {{AI4GSD: an easy-to-use grain size distribution measurement software powered by vision AI}},
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
