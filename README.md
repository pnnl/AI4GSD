# AI4GSD
An end-to-end vision AI tool to map sediment grain size distribution from drone and smartphone photos and videos.

## Introduction
AI4GSD is a computer vision AI tool for automated grain size mapping from videos and photos acquired by drones and smartphones. This tool has been proven to provide highly accurate results for estimating photo resolution (NSE=0.999, mean percent error (MPE) = 0.14%, mean absolute percent error (MAPE) = 0.95%) and median grain size (NSE=0.984, MPE= 0.77%, MAPE=3.50%) for both drone and smartphone images. Per camera type, the accuracy is 0.995, 1.26%, and 2.85% in terms of NSE, MPE, and MAPE for drone images, while the corresponding values are 0.979, 0.34%, and 4.09% for smartphone images. This tool also demonstrates an accuracy of -7.59% and 1.17% when compared to a tutorial example in BaseGrain and field observation data reported in the literature. For each input image, this tool uses three AI models, i.e., Scale AI, Anthro AI, and Grain AI, to detect pixel sizes of reference scales, anthropogenic objects, and individual grains, respectively. The reference scale pixel sizes are then converted to photo resolution (mm/pixel) based on user-defined reference scale sizes. The region occupied by the anthropogenic objects is cropped out as non-grain regions. The pixel sizes of individual grains are converted to real size by multiplying their pixel sizes by the photo resolution. Each photo typically detects hundreds to thousands of grains. Such data is then used to generate a grain size distribution based on count or area weight of each grain. Characteristic grain sizes such as D10, D50, D60, and D84 are then computed from the grain size distribution curve. Such statistical data are output to standard formats for all photos. The tool also outputs the overlay of the AI-predicted reference scale, anthropogenic objects, and individual grains on the raw image for each photo for the user's further analysis. The tool also demonstrates a good computational efficiency, with an average processing time of 2 seconds per frame when figure output is disabled. Interested users are suggested to test relevant functions using the Jupyter Notebook examples.

## Funding
The work is supported by the Program Development Funding from the Physical and Computational Science Directorate (PCSD) of Pacific Northwest National Laboratory (PNNL) and PNNL’s River Corridor Hydrobiogeochemistry Science Focus Area project (https://www.pnnl.gov/projects/river-corridor) funded by US DOE Biological and Environmental Research (BER)

## Reference
Details of the data and model can be found in:

Chen Y ; Renteria L ; Chen R ; Li B ; Forbes B ; Goldman A E (2025): Videos, photos, and AI-derived grain size data associated with “Modular AI and video surveys transform multiscale sediment grain size mapping”. River Corridor Hydro-biogeochemistry from Molecular to Multi-Basin Scales SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/2588483 

Chen, Y., Bao, J., Chen, R., Li, B., Yang, Y., Renteria, L., et al. (2024). Quantifying streambed grain size, uncertainty, and hydrobiogeochemical parameters using machine learning model YOLO. Water Resources Research, 60, e2023WR036456. https://doi.org/10.1029/2023WR036456

Chen Y ; Bao J ; Chen Y ; Li B ; Yang Y ; Stegen J C ; Renteria L ; Delgado D ; Forbes B ; Goldman A E ; Simhan M ; Barnes M ; Laan M ; McKever S A ; Hou Z ; Chen X ; Scheibe T D (2023): Artificial intelligence models, photos, and data associated with the manuscript “Quantifying Streambed Grain Size, Uncertainty, and Hydrobiogeochemical Parameters Using Machine Learning Model YOLO” (v2). River Corridor and Watershed Biogeochemistry SFA, ESS-DIVE repository. Dataset. https://doi.org/10.15485/1999774

## Installation
The following libraries are used, other version of libraries may also work.  

* Python (3.11.13)
* numpy (2.2.6)
* pandas (2.3.1)
* opencv-python (4.12.0)
* matplotlib (3.10.6)
* ultralytics (8.3.186)
* torch (2.9.0+cpu or version fit your CPU/GPU)
  
Using different version of numpy and pandas as listed above may cause error.

## Setup system path
For Windows and Spyder: open Spyder, then go to "Tools>PYTHONPATH manager>add path". Click the "add path (+ marker)", then add the path of the main folder to the system. The main path is a folder which includes "README.MD".

## Run the code
All the codes for the models are in folder AI4GSD. Six Jupyter Notebooks (in Demo_JupyterNotebook) are provided to demonstrate how to use the code. All the necessary data for the demos are in the folder Demo_Data.

* Demo1_Drone_Videos.ipynb : a demo to show how to process video data from drone.
* Demo2_Smartphone_Photos.ipynb : a demo to show how to process photos from smartphones.
* Demo3_Smartphone_Videos.ipynb : a demo to show how to process video data from smartphones.
* Demo4_Smartphone_PhotosAndVideos.ipynb : a demo to show how to process both photos and videos from smartphone.
* Demo5_FieldValidation_BaseGrainExample : a demo to validate the accuracy of the AI model against BaseGrain tutorial example.
* Demo6_FieldValidation_Mair2024SiteK1.ipynb : a demo to validate the accuracy of the AI model against Site K1 field data from Mair D. 2024.

The users can follow the description in each Jupyter Notebook to run the model step by step.

## Cite as

@misc{Chen2025,
address = {Richland, WA, USA},
author = {Chen, Yunxiang},
title = {{AI4GSD: An end-to-end vision AI tool to map sediment grain size distribution from drone and smartphone photos and videos}},
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
