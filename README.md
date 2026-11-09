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

## Installation
Please follow the steps in Install.md to install the software. 

## Setup system path
For Windows and Spyder: open Spyder, then go to "Tools>PYTHONPATH manager>add path". Click the "add path (+ marker)", then add the path of the main folder to the system. The main path is a folder which includes "ReadMe.md".

## Run the code
All the codes for the models are in folder AI4GSD. Six Jupyter Notebooks (in Demo_JupyterNotebook) are provided to demonstrate how to use the code. All the necessary data for the demos are in the folder Demo_Data.

* Demo1_Drone_Videos.ipynb : a demo to show how to process video data from drone.
* Demo2_Smartphone_Photos.ipynb : a demo to show how to process photos from smartphones.
* Demo3_Smartphone_Videos.ipynb : a demo to show how to process video data from smartphones.
* Demo4_Smartphone_PhotosAndVideos.ipynb : a demo to show how to process both photos and videos from smartphone.
* Demo5_FieldValidation_BaseGrainExample : a demo to validate the AI model against BaseGrain tutorial example.
* Demo6_FieldValidation_Mair2024SiteK1.ipynb : a demo to validate the AI model against Site K1 field data from Mair D. 2024.

The users can follow the description in each Jupyter Notebook to run the model step by step.

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