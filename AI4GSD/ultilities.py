#%%######################### Descriptions #####################################
# The code is developed by Dr. Yunxiang Chen @ Pacific Northwest National Lab,#
# Please contact yunxiang.chen@pnnl.gov for feedbacks or questions.           #
# Potential users are permitted to use for non-commercial purposes without    #
# modifying any part of the code. Please contact the author to obtain written #
# permission for commercial or modifying usage.                               # 
#                                                                             #
#                Copyright (c) 2025 Yunxiang Chen                             #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at                                     #
#                                                                             #
#     http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #

#%% Required package.
import os, copy, time, platform, glob, fnmatch, cv2, piexif, shutil, requests,\
    logging,  math, imutils
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from natsort import natsorted, index_natsorted
from pathlib import Path
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

# Loading AI4GSD functions.
from .mathm import __file__ as mathm_path
from .mathm import getFiles, datevec, interp1, etime, center2corners, \
    ismembertol, getFolders, isfile, gpsdist
from .converters import yolo2mask, yolo2txt, yolo2json
from .io import readEXIF, readFC, readSRT, getExif

#%% Default parameters used for Photo2GSD defined in GSD.py file.
class parameters:
    
    # Default parameters.
    defaults = {
        # Parameters for metadata.
        'MinimumFileSize':10,                                                  # Minimum file size considered as valid file.
        'FloatPrecision':'float64',
        'Directory': '',                                                       # Folder path of photos/vidoes.
        'FolderLevel': 'L1',                                                   # Metadata at single folder level.
        'MetadataUnique': False,                                               # Used for multi-folders, removing rows with indentical photo name.
        'ScalesDefault': {'none':-1},                                          # Default photo resolution (m/pixel).
        'Scales': None,                                                        # User defined scales if provied.
        'TemplateVersion':'V3',                                                # Template version.
        'ScaleTemplateName': 'Scales_V3.csv',                                  # A template for photo scale definition.
        'StatisticsTemplateName': 'Statistics_V3.csv',                         # A template for final output statistics data.
        'UserScaleFile':'',                                                    # User-provided scale file.
        'UserGPSFile':'',                                                      # User-provided GPS file.
        'ConfidenceThresholdYOLO':0.25,                                        # Cut off confidence threshold by default in YOLO.
        'ConfidenceThresholdUser':0.35,                                        # Additional cut off confidence threshold by user.
        'YOLOIOU':0.45,                                                        # Default IOU by YOLO.
        'TargetPhotoExtension': ['.jpg','.png','.jpeg','.tif','.tiff'],        # What photo data to process.
        'TargetVideoExtension': ['.mp4','.mov','.avi'],                        # What video data to process.
        'ExcludingPhotos':[''],                                                # What photos to exlude.
        'ModelName':'YOLO11m.1280.20250926',                                   # Nickname of AI model.
        'LandingAIInferenceMethod':'Python',                                   # Method to call LandingAI models.
        'SourceModelName':'',
        'ObjectName':'rock',
        'ImageSizeType':'default',
        'ImageSizeFactor': 1,
        'MaximumNumberOfObject':100_000,
        'YOLOPrint': False,
        'YOLOSave': False,
        'YOLOOverWrite': False,
        'VideoFolderName':'videos',
        'PhotoFolderName':'photos',
        'DroneLogFolderName':'logs',
        'FieldDataFolderName':'field',
        'FieldDataConversionFolderName':'converted',
        'ManualScaleFolderName':'scales_manual',
        'ManualGrainFolderName':'grains_manual',
        'ScaleAIPredictionSaveFolderName':'scales_predicts',
        'GrainAIPredictionSaveFolderName':'grains_predicts',
        'SegmentationSaveFolderName': 'segments',
        'SegmentationSaveJsonOption': 'default',                               # Options: default or labelme.
        'SegmentationClassName': None,                                         # Detect all classes if None.
        'OverWriteSegementation': False,
        'UseMask': True,                                                       # Control if using extral anthropogenic object filter.
        'GrainBackgrouondOverlapThreshold':0.5,                                # Keep photo if background object overlap ratio less than this.
        'InvividualGrainBackgrouondOverlapThreshold':0.5,                      # Keeep invidual grain if its overlap with background oject is less than this.
        'DefaultDate': datetime(1900, 1, 1, 1, 0, 0),
        'NoValueIndicator': -9999.0,
        'CSVEncoding':'latin1',
        'OutputKeyWord':'predict',
        'SaveFigure': True,                                                    # Single control for all save figures.
        'SaveOverlayLabel': True,                                              # Save grain box with image.
        'SaveGSDCurve': True,
        'SaveGSDCurveResolution':300,                                          # Only for grain size distribution curves.
        'SaveOverlayAll': True,                                                # Save grain box, scale, background, and GSD curves.
        'SaveOverlayResolution':900,
        'SaveOverlayGrain':True,
        'SaveOverlayScale':True,
        'SaveOverlaySegmentation':True,
        'SaveOverlayImage': True,
        'SaveOverlayGSD':True,
        'SaveOverlayGSDLegend': True,
        'SaveOverlayManual':True,
        'SaveOverlayField':True,
        'SaveOverlayShowFont': True,
        'SaveOverlayGrainColor': (0,255,0),                                    # Color: green (0,255,0); blue (0,0,255); red (255,0,0); whilte (255,255,255)
        'SaveOverlayScaleColor': (255,0,0),                                    # Color: green (0,255,0); blue (0,0,255); red (255,0,0); whilte (255,255,255)
        'SaveOverlayScaleDotSize': 12,
        'SaveOverlayScaleDotColor': (255,0,0),
        'SaveOverlayBackgroundColor': (0,0,255),                               # Color: green (0,255,0); blue (0,0,255); red (255,0,0); whilte (255,255,255)
        'SegmentationOverlayTransparency': 0.5,
        'SaveYOLOData': True,
        'SaveYOLODataBeforeThresholdFolder': 'BeforeThreshold',
        'SaveObjectData': True,
        'SaveStatisticsData': True,
        'SaveStatisticsVersion':'V1',
        'PrintOnScreen': True,
        'PrintImageSize': False,
        'TemplateFolderName': 'templates',
        'SaveFolder':'',
        'SleepTime':5,
        'ValidationPlotKeyWord':'Validation',
        'UserYOLOFolder':'',
        'OverWriteScaleFile':True,
        'OverWriteScaleRows': True,
        'OverWriteGSDFilePhoto':False,
        'OverWriteGSDFileFolder':False,
        'OverWriteGSDFileSummary':False,
        'OverWriteAll':False,
        'ScaleSource':'default',
        'ComputingSource':'local',
        'DroneResolutionFormula': [-2.59e-5,2.16e-3,0.278,0.0219],
        'ExcludingFolders':'',
        'GSDFlag':'',
        'MaximumGrainSize':1e9,                                                # Unit meter.
        'MinimumGrainSize':0,                                                  # Unit meter.
        'MinimumPhotoResolution': 12_000_000,                                  # Minimum photo resolution.
        'MaximumPhotoResolution': 25_000_000,                                  # Maximum photo resolution.
        'CutGrainSizeMethod': 'wh',
        'ScaleFlag':'',
        'AIScale': {
            'greencap': 0.025,'bluecap': 0.037,'redbluecap':0.025, 
            'green03': 0.00762,'black05': 0.0127,'purple1': 0.0254,
            'blue2': 0.0508,'red4': 0.1016,'yellow8': 0.2032,
            'blue8': 0.2032},
        'OverlayFontColor': 'white',
        'OverlayFontColorForeground': 'black',
        'OverlayFontSizeScaleFactor':2,
        'OverlayFontSize': 2,
        'OverlayLegendFontSize': 2,
        'OverlayGSDLineWidth': 1,
        'OverlayGSDTickLineWidth': 2,
        'OverlayGSDTickMarkLength': 2,
        'OverlayD50MarkerSize': 1.5,
        'OverlayLableLineWidth': 4,                                            # Line width of grain label boxes.
        'OverlayFontLineWidth': 0.5,
        'OverlayFontBoundaryDistanceX': 0.0075,                                # Distance between x labels and photo edge.
        'OverlayFontBoundaryDistanceY': 0.0075,                                # Distance between y labels and photo edge.
        'OverlayLegendAlpha': 0.85,
        'OverlayLegendFrame': True,
        'OverlayLegendShadow': False,                                          # Add 3D effect.    
        'OverlayLegendColor': 'black',
        'OverlayLegendFaceColor': 'white',
        'OverlayLegendEdgeColor': 'none',
        'OverlayLegendAlignment': 'left',
        'OverlayGSDColor': ['brown','blue','black','orange','cyan','red','purple'],
        'OverlayGSDManualColor': ['brown','blue','black','orange','cyan','red','purple'],
        'OverlayGSDFieldColor': ['brown','blue','black','orange','cyan','red','purple'],
        'OverlayGSDLineType': ['-','--',':','-.'],
        'OverlayGSDKey': ['AI4GSD: ','Manual: ','Field: '],
        'OverlayGSDLegend': ['count-width','count-height','count-diagonal',\
                             'area-width','area-height','area-diagonal'],
        'OverlayGSDLegendField': None,
        'OverlayGSDMinSize':None,                                              # Minium grain size in GSD plot.
        'OverlayGSDMaxSize':None,                                              # Maximum grain size in GSD plot.
        'OverlayGSDXTickNumber':6,
        'OverlayGSDYTickNumber':6,
        'OverlayLegendLoc': 'best',
        'OverlayLegendShiftX': None,
        'OverlayLegendShiftY': None,
        'GSDWeight': 'a',                                                      # How to weight grain size: a (area) or c (count).
        'GSDAxis': 'r',                                                        # How to define grain size: w (width), h (height), r (diagnoal).                            
        'SaveVTK': False,
        'SaveVTKRatio': 1,
        'SaveVTKName': 'diagonal_m',
        'SaveVTKNameShow': '\log_2(D_i^r) (mm)',
        'SaveVTKCmap': 'plasma',
        'SaveVTKAlpha': 0.5,
        'SaveVTKFontSize': 12,   
        'CircularityPercentError':5,
        'MinimumGrainNumber':100,
        'MinimumDroneHeight':1.5,                                              # Unit m.
        'Usage':3,                                                             # Data usage: 0 train, 1 val, 2 test, 3 predict.
        'SiteID': 'N/A',
        'Author': 'YC',
        'ZeroReplacedBy':'Z',
        'VideoTimeMode':'first',                                               # Which time to use as video metadata time (first or last).
        }
        
    # Define default parameters and overwrite vlaues if provided by users.
    def __init__(self, **kwargs):
        # Default values.
        for key, default_value in self.defaults.items():
            setattr(self, key, default_value)
        # Overwrite values.
        for key, new_value in kwargs.items():
            setattr(self, key, new_value)
    
    # Define a new variable for better visualization in string format.
    def __repr__(self):
      attr_str = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
      return f"Parameters({attr_str})"
  
    # Convertign the parameters into list for easier post-processing or check.
    def to_dict(self):
      return dict(self.__dict__)
  
#%% Similar to Landing AI default results structure.
class detectionResult:
    
    def __init__(self, label_index, bboxes, score, label_name):
        self.label_index = label_index
        self.bboxes = bboxes                                                   # [xmin, ymin, xmax, ymax]
        self.score = score
        self.label_name = label_name
        
#%% Define the name of the cloud, api, and model id.  5 models available now: #
# RepPoints37M, YOLO11m.1280, YOLO11x.1280, YOLO11m.640, YOLOv5x6u.1280.      #
# More models will be added in the later version.                             #
def cloud(PP):
    
    # Converting parameters.
    model_name = PP.ModelName
    object_name = PP.ObjectName
    
    # Define available api keys for cloud services.
    api_key_landingai = "land_sk_F1yX9I0biyioIpOc9gqLAw2RuK5EOModjmDyritJyA1Okc00oL"
    #api_key_ultralytics = "dc9878dc754243f4db9e39dd0ec73434ca3355c34b"
    api_key_ultralytics = "305af8455bb42e20e7a2f9915f49889c52c890b8ab"

    # Define api key and model id for each model.
    if object_name.lower() == 'rock':                                          # For rock detection.
        if model_name.lower() == 'reppoints37m':
            cloud_name = 'landingai'
            api_key = api_key_landingai
            model_id = "5b9af769-a339-4502-9acd-5e5a9c1aa5a3"                  # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.1280.20250127':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-x8dp4dnfw9cyd1tvsrao-7nza6zqsha-uw.a.run.app"     # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.1280.20250307':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-tg3tve71gqfwclffm8v5-7nza6zqsha-uw.a.run.app"     # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.1280.20250308':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-ldb3uhz1i0hvr96lmkrr-7nza6zqsha-uw.a.run.app"     # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.1280.20250309':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-lg8r1hyi1fzodwoopwxi-7nza6zqsha-uw.a.run.app"     # Rock67.40.6.21 (with drone photos)
        elif model_name.lower() == 'yolo11m.1280.20250319':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-yze4ajzlqchewrwsnfhy-7nza6zqsha-uw.a.run.app"     # Rock48.42.6.0 (with 6 drone photos)
        elif model_name.lower() == 'yolo11m.1280.20250320':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-hpmyaulbe04iagrjtt2r-7nza6zqsha-uw.a.run.app"     # Rock48.42.6.0 (with 6 drone photos, train 200 steps)
        elif model_name.lower() == 'yolo11m.1280.20250321':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-gpss4dimpjhxrj8eozvt-7nza6zqsha-uw.a.run.app"     # Rock53.47.6.0 (with 6 drone photos and 41 smartphone photos, train 150 steps)
        elif model_name.lower() == 'yolo11m.1280.20250322':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-mgva2halm73fiudhtx2x-7nza6zqsha-uw.a.run.app"     # Rock53.47.6.0 (with 6 drone photos and 41 smartphone photos, train 150 steps)
        elif model_name.lower() == 'yolo11m.1280.20250926':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-jowz8r2y0ngz3vrv25mb-7nza6zqsha-uw.a.run.app"     # Rock54.47.7.0 (with 6 drone photos and 41 smartphone photos, train 500 steps, resized photos)
        elif model_name.lower() == 'yolo11m.2560.20250927':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-xct4h6inyykufoucdbeo-7nza6zqsha-uw.a.run.app" 
        elif model_name.lower() == 'yolo11x.1280.20250320':                                                     
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            "https://predict-1pkqfj3y5qsr1pjfv7c9-7nza6zqsha-uw.a.run.app"     # Rock48.42.6.0 (with 6 drone photos, train 200 steps, YOLO11x)
        elif model_name.lower() == 'yolo11x.1280':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-dtyttein6qxmemrmargh-7nza6zqsha-uw.a.run.app'     # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.640':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-gblklsygdhtnxxr5dkro-7nza6zqsha-uw.a.run.app'     # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.640.20250307':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-pobpogyup0psbimqvdjb-7nza6zqsha-uw.a.run.app'     # Rock61.36.5.20
        elif model_name.lower() == 'yolo11m.640.20250309':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-ea4ypgsr0azddntfbsxc-7nza6zqsha-uw.a.run.app'     # Rock67.40.6.21 (with drone photos)
        elif model_name.lower() == 'yolov5x6u.1280':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-diwv6eohhcbywp8b7es2-7nza6zqsha-uw.a.run.app'     # Rock61.36.5.20
        else:
            cloud_name = ''; api_key = ''; model_id = ''                       # Set empty if invalid model provided.
            if PP.UserYOLOFolder == '': 
                raise 'Invalid model name'
    elif object_name.lower() == 'cap':
        if model_name.lower() == 'yolo11m.1280.20250518':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-dijwyt60ldtbimtz1pbm-7nza6zqsha-uw.a.run.app'     # Cap72.50.7.15.20
        elif model_name.lower() == 'yolo11m.1280.20250825':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-62zeqhkpuwrcrbrwgefb-7nza6zqsha-uw.a.run.app'     # Cap117.80.11.26
        else:
            cloud_name = ''; api_key = ''; model_id = ''                       # Set empty if invalid model provided.
            if PP.UserYOLOFolder == '': 
                raise 'Invalid model name'
    elif object_name.lower() == 'anthro':
        if model_name.lower() == 'yolo11m.640.20250527':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://hub.ultralytics.com/models/LANJV1QBekDh5wGntsli'          # Anthro.x.x.x.x.
        elif model_name.lower() == 'yolo11m.1280.20250906':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-sk3ztnmguv3df8bbe0qb-7nza6zqsha-uw.a.run.app'     # Anthro.256.179.26.51 (1280)
        elif model_name.lower() == 'yolo11m.640.20250906':
            cloud_name = 'ultralytics'
            api_key = api_key_ultralytics
            model_id = \
            'https://predict-dvtvldi908edrspejgdb-7nza6zqsha-uw.a.run.app'     # Anthro.256.179.26.51 (640)
        else:
            cloud_name = ''; api_key = ''; model_id = ''                       # Set empty if invalid model provided.
            if PP.UserYOLOFolder == '': 
                raise 'Invalid model name'
    else:
        raise 'Invalid detection object'
        
    return cloud_name, api_key, model_id

#%% Define the absolute path of the pre-trained AI models based on user       #
# parameters. These models can run locally without accessing cloud, which     #
# will be much faster. More models will be added in newer versions.           #
def local(PP):
    # Define api key and model id for each model.
    cloud_name = ''; api_key = ''; model_id = ''                               # Set empty if invalid model provided.
    
    # Parent folder path where pre-trained AI models sit.
    exe_path = os.path.dirname(mathm_path) + os.sep + 'exes'
    
    # Converting parameters.
    model_name = PP.ModelName
    object_name = PP.ObjectName
    
    # Define api key and model id for each model.
    if object_name.lower() == 'rock':                                          # For rock detection.
        if model_name.lower() == 'yolo11m.1280.20250322':                                                     
            ai_model = 'rockai.47.7.0.1280.yolo11m.20250322.pt'                # Rock54.47.7.0 (with 6 drone photos and 41 smartphone photos, train 150 steps)
        elif model_name.lower() == 'yolo11m.1280.20250926':                                                     
            ai_model = 'rockai.47.7.0.1280.yolo11m.20250926.pt'                # Rock54.47.7.0 (with 6 drone photos and 41 smartphone photos, train 500 steps)
        elif model_name.lower() == 'yolo11m.2560.20250927':                                                     
            ai_model = 'rockai.47.7.0.2560.yolo11m.20250927.pt'                # Rock54.47.7.0 (with 6 drone photos and 41 smartphone photos, train 300 steps)
        else:
            raise 'Invalid model name'
    elif object_name.lower() == 'cap':
        if model_name.lower() == 'yolo11m.1280.20250518':
            ai_model = 'capai.50.7.15.1280.yolo11m.20250518.pt'                # Cap72.50.7.15.20
        elif model_name.lower() == 'yolo11m.1280.20250825':
            ai_model = 'capai.80.11.26.1280.yolo11m.20250825.pt'               # Cap117.80.11.26
        else:
            raise 'Invalid model name'
    elif object_name.lower() == 'anthro':
        if model_name.lower() == 'yolo11m.640.20250527':
            ai_model = 'anthroai.0.0.0.640.yolo11m.20250527.pt'                # YOLO default segmentation model.
        elif model_name.lower() == 'yolo11m.1280.20250906':
            ai_model = 'anthroai.179.26.51.1280.yolo11m.20250906.pt'           # YOLO default segmentation model.
        elif model_name.lower() == 'yolo11m.640.20250906':
            ai_model = 'anthroai.179.26.51.640.yolo11m.20250906.pt'           # YOLO default segmentation model.
        else:
            raise 'Invalid model name'
    else:
        raise 'Invalid detection object'
    
    # Absolute path of the AI model.
    model_id = exe_path + os.sep + ai_model
    
    return cloud_name, api_key, model_id

#%% Define the name of the cloud, api, and model id.  5 models available now: #
# RepPoints37M, YOLO11m.1280, YOLO11x.1280, YOLO11m.640, YOLOv5x6u.1280.      #
# More models will be added in the later version.                             #
def agent(PP):
    if PP.ComputingSource.lower() == 'cloud':
        cloud_name, api_key, model_id = cloud(PP)
    elif PP.ComputingSource.lower() == 'local':
        cloud_name, api_key, model_id = local(PP)
    else:
        raise 'Invalid computing type'
        
    return cloud_name, api_key, model_id

#%% Extracting metadata of L1 folder which contains both photos and videos.   #
# Notes on drone videoes, photoes, and video-derivied images: 1) Time and GPS #
# in metadata are most accurate when both flight logs and video SRT data are  #
# available; 2) If both flight record and SRT are available, the video time is#
# the time of frame 1 in SRT minus 1/frame rate (estimated time for frame 0); #
# 3) For drone video-drived images, the resutls are from 4 mode: 3a) with both#
# logs and srt; 3b) with logs but no srt; 3c) no logs but with srt;3d) no logs#
#, no srt. 3a is most accurate; 3b == 3d; 3c is less accurate due to missing  #
# logs. 4) For smartphone video images:  no srts, no logs, equivalent to 3d.  #
def metadataLevel1(PP):
    
    #t0 = time.time()
    # Print information control.
    formatSpec1 = '%s/%s: video %s/%s, %s (%3.2f%%), %3.2f s'
    formatSpec2 = '%s/%s: photo %s/%s, %s, scale %s, res %3.2f mm/px, status %s, (%3.2f%%), %3.2f s'
    formatSpec2a = '%s/%s: photo %s/%s, %s, scale %s, res %d px/px, status %s, (%3.2f%%), %3.2f s'
    formatSpec2b = 'Model layer: %s, model parmeter: %4.2fM, submodule number: %s'
    formatSpec2c = 'IOU: %3.2f, user confidence: %3.2f'
    #formatSpec3 = 'Total time %3.2f s, time per data %.2f s'
    #formatSpec4= 'Total time %3.2f s'
    
    # Checking if working diretory has been specified.
    if PP.Directory == '': raise ValueError('Working directory is not defined.')
    
    # Pass parameters from PP to local.
    PP.ObjectName = 'cap'
    DirectoryL1 = PP.Directory
    SCUserFile= PP.UserScaleFile
    #GPSFile= PP.UserGPSFile
    OverWriteSC = PP.OverWriteScaleFile
    OverWriteSCRows = PP.OverWriteScaleRows
    OverWriteAll = PP.OverWriteAll 
    scsource = PP.ScaleSource.lower()
    printimgsize = PP.PrintImageSize
    SCName= PP.ScaleTemplateName
    refdate = PP.DefaultDate
    zmax = PP.NoValueIndicator
    outkeyword = PP.OutputKeyWord
    CSVEncoding = PP.CSVEncoding
    pext =  PP.TargetPhotoExtension
    vext = PP.TargetVideoExtension
    validationkeyword = PP.ValidationPlotKeyWord
    model_name = PP.ModelName
    scalefactor = PP.ImageSizeFactor                                           # Used to control image size for inference.
    sc = PP.AIScale                                                            # Default sizes of reference scales.
    poly = np.poly1d(PP.DroneResolutionFormula)                                # The calibrated resolution formula of drone.
    scflag = PP.ScaleFlag
    pr = PP.FloatPrecision
    iou = PP.YOLOIOU
    conf1 = PP.ConfidenceThresholdUser
    stid = PP.SiteID
    zeroby = PP.ZeroReplacedBy
    tdir = PP.TemplateFolderName
    vdir = PP.VideoFolderName
    pdir = PP.PhotoFolderName
    ddir = PP.DroneLogFolderName
    mdir = PP.ManualScaleFolderName
    sdir = PP.ScaleAIPredictionSaveFolderName
    
    # Defining cloud information.
    cloud_models=['RepPoints37M','YOLO11m.1280.20250127','YOLO11x.1280',
                  'YOLO11m.640','YOLOv5x6u.1280','YOLO11m.1280.20250307',
                  'YOLO11m.640.20250307','YOLO11m.1280.20250308',
                  'YOLO11m.640.20250309','YOLO11m.1280.20250309',
                  'YOLO11m.1280.20250319','YOLO11m.1280.20250320',
                  'YOLO11x.1280.20250320','YOLO11m.1280.20250321',
                  'YOLO11m.1280.20250322','YOLO11m.1280.20250518',
                  'YOLO11m.1280.20250825']
    
    # Raise error if folder does not exist.
    if not os.path.exists(DirectoryL1):
        raise ValueError(f"{DirectoryL1} does not exist")
    
    # Loading requried scale template file and extract its column names.
    codedir = os.path.dirname(mathm_path) + os.sep + tdir
    SCFile = os.path.join(codedir, SCName)
    if os.path.isfile(SCFile):
        headers = pd.read_csv(SCFile,keep_default_na=False)
        cnames = list(headers.columns)
        idx1a = cnames.index('L1_px1_pixel')
        idx1b = cnames.index('Duration_second')
        idx2a = cnames.index('Make')
        idx2b = cnames.index('Notes')
        idx3 = cnames.index('Site_ID')
        defaults = ['N/A']*3 + [zmax]*(idx1b - idx1a + 1) +\
            [refdate] + [zmax] + ['N/A']*(idx2b - idx2a + 1)
        defaults[idx3] = stid
        [SCNameStem,ext] = SCName.split('.')
    else:
        raise ValueError(f"Missing required scale template file {SCFile}.")
        
    # Loading user-defined scale files if existed.
    SCTable = pd.DataFrame()                                                   # default empty    
    if os.path.isfile(SCUserFile):
        SCTable = pd.read_csv(SCUserFile,keep_default_na=False)
        SCTable = SCTable.apply(pd.to_numeric, errors="coerce").\
            astype(pr).combine_first(SCTable)
        SCTable['Name_stem'] = SCTable['Name'].apply(lambda x: Path(str(x)).stem)
        
    # Loading GPS data if existed.
    # SiteID = []
    # SiteGPS = []
    # if os.path.isfile(GPSFile):
    #     GPSTable = pd.read_csv(GPSFile, encoding= CSVEncoding,keep_default_na=False)
    #     SiteID = GPSTable.iloc[:,0].astype(str).tolist()
    #     SiteGPS = GPSTable.iloc[:,1:3].values
        
    # Obtaining the parement directory.
    one_dir = os.path.basename(DirectoryL1)
    print(f"Generating scale file for: {one_dir}")
    fnames = one_dir.split('_')
    author = fnames[1] if len(fnames) >1 else PP.Author
    note0 = fnames[2] if len(fnames) >2 else 'N/A'
    usage = PP.Usage
    d = one_dir.lower()
    if usage is None:
        usage = 3
        if d.startswith('train'):
            usage = 0
        elif d.startswith('val') or d.startswith('validate') or d.startswith('validation'):
            usage = 1
        elif d.startswith('test'):
            usage = 2
        
    # Define output scale file name and check if it exists.
    if scflag == '':
        snamexls=DirectoryL1+os.sep+SCNameStem+'_'+one_dir+ '.' + ext
    else:
        snamexls=DirectoryL1+os.sep+SCNameStem+'_'+one_dir+ '_' + scflag + '.' + ext
    key1 = 'Name'
    key2 = 'L1L2_res_meter_per_pixel'
    scales_df0 = pd.DataFrame({key1: [''],key2: [-1]})                         # Set default photo resolution as -1 m/pixel.
    if os.path.isfile(snamexls) and os.path.getsize(snamexls) >10:    
        scales_df0 = pd.read_csv(snamexls,encoding= CSVEncoding,\
                                 keep_default_na=False)                        # Loading existing scale file if existed.
            
    if os.path.isfile(snamexls) and (not OverWriteSC) and (not OverWriteAll):
        print("Scale file existed, skip...")
        scales_df = scales_df0
    else:
        scales_df = pd.DataFrame()
        names,stems,exts,ispv = getFiles(DirectoryL1,vext=vext,pext = pext)
        
        # For all videos.
        count = 0
        v_names = []; v_stems = []; v_exts = []
        for namei, stemi, exti, ispvi in zip(names, stems, exts, ispv):
            if ispvi == 1:
                v_names.append(namei)
                v_stems.append(stemi)
                v_exts.append(exti)
        idx = index_natsorted(v_names)
        v_names = np.array(v_names)[idx]
        v_stems = np.array(v_stems)[idx]   
        nvids = len(v_names)
        
        # All image file names.
        img_names = []; img_stems = []; img_exts = []
        for namei, stemi, exti, ispvi in zip(names, stems, exts, ispv):
            if ispvi == 0 and outkeyword not in stemi.split('_') and \
                validationkeyword not in stemi.split('_'):
                img_names.append(namei)
                img_stems.append(stemi)
                img_exts.append(exti)                
        idx = index_natsorted(img_names)
        img_names = np.array(img_names)[idx]
        img_stems = np.array(img_stems)[idx]
        npics = len(img_names)
        nvt = nvids + npics
        
        # Loading STR data if exists.
        #VGPS = []
        status = -1
        if nvids == 0:
            #print("No video found, skipped.")
            scales_v = pd.DataFrame(columns=cnames)                            # Save data for video metadata.
        else:
            v_keywords = [s.split('_')[0] for s in v_stems]
            v_sites = [s.split('_')[1] for s in v_stems]
            psize = np.zeros((nvids, 2), dtype=int)
            orientation = np.zeros((nvids,), dtype=int)
            record_list = []
            defaults_dict = dict(zip(cnames, defaults))
            
            # Loading drone flight data if availabel.
            FCPath = os.path.join(DirectoryL1, ddir)
            fct = []
            fcv_path = ''
            if 'DJI' in v_keywords and os.path.isdir(FCPath):
                log_path = natsorted(list(Path(FCPath).glob('*Airdata.csv')))
                fc_path = natsorted(list(Path(FCPath).glob('*FC*.csv')))
                if len(log_path)>0 and len(fc_path)>0:
                    print('Reading DJI flight logs from files')
                    fcv_name = '_'.join(log_path[0].stem.split('-')[:4]) + '_FCV.csv'
                    fcp_name = '_'.join(log_path[0].stem.split('-')[:4]) + '_FCP.csv'
                    fcv_path = Path(log_path[0].parent,fcv_name)
                    fcp_path = Path(log_path[0].parent,fcp_name)
                    fcv = pd.read_csv(fcv_path)
                    lognames = fcv.columns.to_list()
                    fct.append(fcv.to_numpy())
                    del fcv
                    fct.append(pd.read_csv(fcp_path).to_numpy())
                    fcv_path = str(fcv_path)
                elif len(log_path)>0 and len(fc_path)==0:
                    print('Loading flight logs for DJI')
                    fct, _, lognames = readFC(FCPath)
                    fcv_name = '_'.join(log_path[0].stem.split('-')[:4]) + '_FCV.csv'
                    fcv_path = str(Path(log_path[0].parent,fcv_name))   
                    
            srts = {}
            ttv = zmax*np.ones((nvids,6))
            for j in range(nvids):
                count = count + 1
                t_start = time.time()
                v_file = v_names[j]
                v_stem = v_stems[j]
                srt_path = os.path.join(DirectoryL1, v_stem + ".SRT")
                vpath = os.path.join(DirectoryL1, v_file)
     
                # Get EXIF.
                exif_info = readEXIF(vpath)
                w = exif_info['Width']
                h = exif_info['Height']
                orient = exif_info['Orientation']
                mak = exif_info['Make']
                mdl = exif_info['Model']
                fr = exif_info['FrameRate']
                dr = exif_info['Duration']
                note = note0
                uxy, ux, uy, uz = [zmax] * 4
                
                # Extracting lat/lon from gps_info if available3
                lat_deg, lon_deg, alt, alt_ref = zmax, zmax, zmax, zmax
                if exif_info['GPSInfo'] is not None:
                    gps_info = exif_info['GPSInfo']
                    lat_deg = gps_info['GPSLatitude']
                    lon_deg = gps_info['GPSLongitude']
                    alt = gps_info['GPSAltitude']
                    alt_ref = gps_info['GPSAltitudeRef'] 
                dtm = exif_info['DateTime']                                    # a datetime obj or None
                if dtm is None: dtm = refdate
                dtm_str = dtm.strftime("%Y-%m-%d %H:%M")
                tsec = dtm.second
                ttv[j,0:6] = datevec(dtm)                                      # Video metadata time.                  
                
                # Update video gps and time from flight record if available.
                if lat_deg == zmax or lon_deg == zmax:
                    # Read video log data *.SRT if not previously read.
                    if v_stem not in srts:
                        DirectoryL1V =  DirectoryL1 + os.sep
                        srt_path = DirectoryL1V + os.sep + v_stem + '.SRT'
                        if os.path.isfile(srt_path):
                            if fcv_path == '':
                                print('SRT is not aligned with flight log data...')
                            _, srtj = readSRT(srt_path,fcv_path)
                            srts[v_stem] = srtj
                    
                    # Obtain more accuracy time information from the SRT.
                    if v_stem in srts:
                        gpst01 = srts[v_stem][[0,-1],9:]                       # Time: YMDHMS, ms, us, lat, lon, height.
                        gpst = gpst01[0,:]
                        ttv[j,0:6] = gpst[0:6]                                 # YMDHM (no seconds).
                        ttv[j,5] = gpst[5]+gpst[6]/1e3+gpst[7]/1e6 - 1*1/fr    # Estimated time for frame 0.
                        
                        # Update gps and time.
                        lat_deg = gpst[8]
                        lon_deg = gpst[9]
                        tsec =  ttv[j,5]
                        y, m, d, H, M = ttv[j,0:5].astype(int)
                        if tsec<0:
                            dtm_new = datetime(y, m, d, H, M, 0)
                        else:
                            dtm_new = datetime(y, m, d, H, M, int(tsec))
                        dtm_str = dtm_new.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Identifying speed from fct if availabe.
                        if len(fct)>0:
                            tt0 = fct[0][0,1:7]
                            dtt = etime(tt0,ttv[j,:])
                            logtargets = ['xSpeed(mps)','ySpeed(mps)','zSpeed(mps)']
                            idx = lognames.index(logtargets[0])
                            idy = lognames.index(logtargets[1])
                            idz = lognames.index(logtargets[2])
                            ux = interp1(fct[0][:,0],fct[0][:,idx],dtt)[0]
                            uy = interp1(fct[0][:,0],fct[0][:,idy],dtt)[0]
                            uz = interp1(fct[0][:,0],fct[0][:,idz],dtt)[0]
                            #uxy = np.sqrt(ux**2+uy**2)
                            
                            gpst01c = gpst01.copy()
                            gpst01c[:,5] = gpst01[:,5] + gpst01[:,6]/1e3 + gpst01[:,7]/1e6
                            dtt01 = etime(tt0,gpst01c[:,0:6])
                            uxa = fct[0][(fct[0][:,0]>=dtt01[0]) & (fct[0][:,0]<=dtt01[1]),idx]
                            uya = fct[0][(fct[0][:,0]>=dtt01[0]) & (fct[0][:,0]<=dtt01[1]),idy]
                            uxya = np.sqrt(uxa **2 + uya **2)                  
                            uxy = np.mean(uxya)                                # Average horizontal speed.                    
                            
                # Get video index.
                vidx = v_sites[j]
                if vidx.isdigit():
                    n0 = len(vidx) - len(vidx.lstrip('0'))
                    if n0>0: vidx = zeroby * n0 + vidx[n0:]
                v_sites[j] = vidx
                
                # Combining all data in a single list.
                update_list = {
                    "Name": v_file,
                    "Folder": one_dir,
                    "Scale_source": "N/A",
                    "Width_pixel": w,
                    "Height_pixel": h,
                    "L1L2_res_meter_per_pixel": -1,                            # Default resolution -1 m/pixel.
                    "L1L2_res_millimeter_per_pixel": -1,                       # Default resolution -1 m/pixel.
                    "Valid": 1,
                    "Corrected": 0,
                    "Usage": usage,
                    "Photo_indicator": 0,                                      # 0 for video, 1 for photo.
                    "Latitude": lat_deg,
                    "Longitude": lon_deg,
                    "Altitude_meter": alt,
                    "Altitude_reference": 0,
                    "Height_m": zmax,
                    'Speed_meter_per_second': uxy,
                    'Speed_x_meter_per_second': ux,
                    'Speed_y_meter_per_second': uy,
                    'Speed_z_meter_per_second': uz,
                    'Frame_rate_frames_per_second': fr,
                    'Duration_second': dr,
                    "Date_PST": dtm_str,
                    "Date_Second": tsec,
                    "Make": mak,
                    "Model": mdl,
                    "Author": "YC",
                    "Keyword": v_keywords[j],
                    "Site_ID": v_sites[j] if stid == 'N/A' else stid,
                    "Video_ID": vidx,
                    "Notes": note
                    }
                
                defaults_dict_new = defaults_dict.copy()
                defaults_dict_new.update(update_list)
                record_list.append(defaults_dict_new)
                    
                t_end = time.time()
                dt = t_end - t_start
                print(formatSpec1 %(count, nvt, j+1,nvids, v_file, 100*(j+1)/nvids, dt))

            # Convert the list to a DataFrame
            scales_v = pd.DataFrame(record_list, columns=cnames)
    
            # Merge with any existing SCTable entries, etc. 
            if not SCTable.empty:
                scales_v_stems = scales_v["Name"].apply(lambda x: os.path.splitext(x)[0])
                SCTable_stems = SCTable["Name"].apply(lambda x: os.path.splitext(x)[0])
                
                # Updating the new scales from existing scale information.
                for idx_v in scales_v.index:
                    stem_v = scales_v_stems[idx_v]
                    VideoExtractedPhoto = False
                    if 'DJI' in stem_v and stem_v.count('_') == 3:
                        stem_v_new = "_".join(stem_v.split("_")[:-1])
                        matches = SCTable_stems == stem_v_new
                        VideoExtractedPhoto = True
                    else:
                        matches = SCTable_stems == stem_v
                        
                    if matches.any(): 
                        idx_sc = min(matches[matches].index.tolist())
                        
                        # Skip photo which has similar video extracted photo names.
                        if SCTable.loc[idx_sc,'Photo_indicator'] == 1 \
                            and VideoExtractedPhoto: continue
                    
                        for cname in cnames:
                            # For datatime format.
                            if cname in 'Date_PST':
                                if scales_v.loc[idx_v,cname] == refdate:
                                    scales_v.loc[idx_v,cname] = \
                                        SCTable.loc[idx_sc,cname]
                            # For string formats.
                            elif cname in ['Folder','Scale_source','Make', 'Model',\
                                           'Author','Keyword','Site_ID','Video_ID','Notes']:
                                if scales_v.loc[idx_v,cname].lower() == 'n/a':
                                    scales_v.loc[idx_v,cname] = \
                                        SCTable.loc[idx_sc,cname]
                            elif cname in ['Name']:
                                1 
                            # For numerical formats.
                            else:
                                scales_v.loc[idx_v,cname] = \
                                    SCTable.loc[idx_sc,cname]
                                    
        # Now we handle images
        if npics == 0:
            scales_p = pd.DataFrame(columns=cnames)
        else:
            # Loading flight path log files drone data if existed.
            keywords = [s.split('_')[0] for s in img_stems]
            sites = [s.split('_')[1] if '_' in s else "N/A" for s in img_stems]
            FCPath = os.path.join(DirectoryL1, ddir)
            fct = []
            fcv_path = ''
            if 'DJI' in keywords and os.path.isdir(FCPath):
                log_path = natsorted(list(Path(FCPath).glob('*Airdata.csv')))
                fc_path = natsorted(list(Path(FCPath).glob('*FC*.csv')))
                if len(log_path)>0 and len(fc_path)>0:
                    print('Reading DJI flight logs from files')
                    fcv_name = '_'.join(log_path[0].stem.split('-')[:4]) + '_FCV.csv'
                    fcp_name = '_'.join(log_path[0].stem.split('-')[:4]) + '_FCP.csv'
                    fcv_path = Path(log_path[0].parent,fcv_name)
                    fcp_path = Path(log_path[0].parent,fcp_name)
                    fcv = pd.read_csv(fcv_path)
                    lognames = fcv.columns.to_list()
                    fct.append(fcv.to_numpy())
                    del fcv
                    fct.append(pd.read_csv(fcp_path).to_numpy())
                    fcv_path = str(fcv_path)
                elif len(log_path)>0 and len(fc_path)==0:
                    print('Loading flight logs for DJI')
                    fct, _, lognames = readFC(FCPath)
                    
            # Output AI model information if enabled.
            if scsource in 'ai':
                if any(model_name.lower() == item.lower() for item in cloud_models):
                    cmpt = PP.ComputingSource.lower()
                    cloud_name, api_key, model_id = agent(PP)
                    print(f'Estimating photo resolution via {cmpt} AI model: {model_name}')
                    
                    # Loading YOLO model
                    if cmpt in 'local': 
                        from ultralytics import YOLO
                        model = YOLO(str(Path(model_id)))
                        logging.getLogger("ultralytics").setLevel(logging.ERROR)
                        
                        import torch.nn as nn
                        modules = list(model.model.modules())
                        nmodules = len(modules)
                        npp = sum(p.numel() for p in model.model.parameters())/1e6
                        nl = sum(isinstance(m, nn.Conv2d) for m in modules)
                        print(formatSpec2b %(nl, npp,nmodules))
                    print(formatSpec2c %(iou,conf1))

            psize = np.zeros((npics, 2), dtype=int)
            orientation = np.zeros((npics,), dtype=int)
            record_list = []
            defaults_dict = dict(zip(cnames, defaults))
            
            # gps 0-1: lat (decimal); lon (decimal)
            # gps 2-3: altitude and ref.
            # gps 4-5: lat and lon from P.GPSFileName.
            # gps 6-10: heights from manual, sonar, above drone, above
            # takeoff, velocity integrate. unit: meter.
            # gps 11-15: scale from  manual, sonar, above drone, above
            # takeoff, velocity integrate.
            gps = zmax*np.ones((npics,4+2+5*2))
            tt = zmax*np.ones((npics,6))
            srts = {}
            for j in range(npics):
                count = count + 1
                t_start = time.time()
                pic_name = img_names[j]
                pic_stem = pic_name.split('.')[0]
                pic_path = os.path.join(DirectoryL1, pic_name)
                note = note0
                vidx = sites[j]
                if vidx.isdigit():
                    n0 = len(vidx) - len(vidx.lstrip('0'))
                    if n0>0: vidx = zeroby * n0 + vidx[n0:]
                sites[j] = vidx

                fr, dr, uxy, ux, uy, uz = [zmax] * 6
                rowj = scales_df0[scales_df0[key1] == pic_name]                # Checking if scale data exists for pic_name
                if not(rowj.shape[0]>0 and rowj[key2].values[0]>0) or \
                    OverWriteSCRows or OverWriteAll:
                        
                    # Read default EXIF info for all photos.
                    exif_info = readEXIF(pic_path)
                    w = exif_info['Width']
                    h = exif_info['Height']
                    fr = exif_info['FrameRate']
                    dr = exif_info['Duration']                    
                    mak = exif_info['Make']
                    mdl = exif_info['Model']
                    orient = exif_info['Orientation']
                    if orient is not None and orient > 4:
                        w, h = h, w
                    psize[j, 0] = w
                    psize[j, 1] = h
                    orientation[j] = orient if orient else 1
                    lat_deg, lon_deg, alt, alt_ref = zmax, zmax, zmax, zmax
                    if exif_info['GPSInfo'] is not None:
                        gps_info = exif_info['GPSInfo']
                        lat_deg = gps_info.get('GPSLatitude', 0)
                        lon_deg = gps_info.get('GPSLongitude', 0)
                        alt = gps_info.get('GPSAltitude', 0)
                        alt_ref = gps_info.get('GPSAltitudeRef',0)
                    dtm = exif_info['DateTime']                                # a datetime obj or None
                    if dtm is None: dtm = refdate
                    tt[j,:] = datevec(dtm)         
                    gps[j,0] = lat_deg
                    gps[j,1] = lon_deg
                    gps[j,2] = alt
                    gps[j,3] = alt_ref
                    
                    # Dealing with photos from drone or drone video frames.
                    ispng = pic_name.lower().endswith('.png')
                    isdrone = 'dji' in pic_stem.lower()
                    picstr = pic_stem.split('_')
                    pv_indiactor = 2 if len(picstr) == 3 else 1
                    if ispng:
                        if len(picstr) == 2:
                            # Update time and GPS from drone image metadata.
                            exif_found = False
                            DirectoryL1P =  DirectoryL1 + os.sep + pdir
                            for ext in ['.jpg','.jpeg','.tiff']:
                                pic_path2=DirectoryL1P +os.sep + pic_stem + ext
                                if isfile(pic_path2): exif_found = True; break 
                            if exif_found: 
                                exif_info = readEXIF(pic_path2)
                                mak = exif_info['Make']
                                mdl = exif_info['Model']
                                lat_deg, lon_deg, alt, alt_ref = [zmax]*4
                                if exif_info['GPSInfo'] is not None:
                                    gps_info = exif_info['GPSInfo']
                                    lat_deg = gps_info.get('GPSLatitude', 0)
                                    lon_deg = gps_info.get('GPSLongitude', 0)
                                    alt = gps_info.get('GPSAltitude', 0)
                                    alt_ref = gps_info.get('GPSAltitudeRef',0)
                                dtm = exif_info['DateTime']                    # a datetime obj or None
                                if dtm is None: dtm = refdate
                                tt[j,:] = datevec(dtm)         
                                gps[j,0] = lat_deg
                                gps[j,1] = lon_deg
                                gps[j,2] = alt
                                gps[j,3] = alt_ref
                                
                        elif len(picstr) == 3:
                                # Read video exif info from video metadata.
                                DirectoryL1V =  DirectoryL1 + os.sep + vdir
                                vid_stem = "_".join(pic_name.split("_", 2)[:2])
                                fmid = float(pic_stem.split("_", 2)[2])
                                pic_path2 = DirectoryL1V + os.sep + vid_stem + '.mov'
                                # Time derived from video file.
                                if isfile(pic_path2):
                                    exif_info = readEXIF(pic_path2)
                                    mak = exif_info['Make']
                                    mdl = exif_info['Model']
                                    lat_deg, lon_deg, alt, alt_ref = [zmax]*4
                                    if exif_info['GPSInfo'] is not None:
                                        gps_info = exif_info['GPSInfo']
                                        lat_deg = gps_info.get('GPSLatitude', 0)
                                        lon_deg = gps_info.get('GPSLongitude', 0)
                                        alt = gps_info.get('GPSAltitude', 0)
                                        alt_ref = gps_info.get('GPSAltitudeRef',0)
                                    dtm = exif_info['DateTime']                # a datetime obj or None
                                    if dtm is None: dtm = refdate
                                    tt[j,:] = datevec(dtm)         
                                    gps[j,0] = lat_deg
                                    gps[j,1] = lon_deg
                                    gps[j,2] = alt
                                    gps[j,3] = alt_ref
                                    
                                    # Identify frame index and compute time at this frame.
                                    dr = exif_info['Duration']
                                    fr = exif_info['FrameRate']
                                    tsec = fmid*1/fr if fr>0 else 0            # How many seconds of the photo.
                                    if dtm != refdate:
                                        dtm = dtm + timedelta(seconds=tsec)
                                        tt[j,:] = datevec(dtm)
                                        
                                # Read video log data *.SRT if not previously read.
                                if vid_stem not in srts:
                                    srt_path = DirectoryL1V + os.sep + vid_stem + '.SRT'
                                    if os.path.isfile(srt_path):
                                        #if os.path.isfile(gps_path):
                                        #    srtj = pd.read_csv(gps_path).to_numpy()
                                        #else:
                                        if fcv_path == '':
                                            print('SRT is not aligned with flight log data...')
                                        _, srtj = readSRT(srt_path,fcv_path)
                                        srts[vid_stem] = srtj
                                        
                                # Obtain more accurate GPS and time info from the SRT.
                                # Time derived from SRT files.
                                if vid_stem in srts:
                                    fmid_max = srts[vid_stem][-1,0]
                                    if fmid<=fmid_max:
                                        gpst = interp1(
                                            srts[vid_stem][:,0], 
                                            srts[vid_stem][:,9:],fmid,
                                            method='linear')
                                    else:
                                        gpst = srts[vid_stem][-1,9:]
                                        t_last = pd.Timestamp(*map(int, gpst[:6]))
                                        ts_new = gpst[6]/1e3 + gpst[7]/1e6 + (fmid - fmid_max)/fr
                                        t_new = t_last +  pd.to_timedelta(ts_new,unit='s')
                                        gpst[:6] = np.array(list(t_new.timetuple()[:6]))
                                        gpst[6] = t_new.microsecond//1000
                                        gpst[7] = t_new.microsecond%1000
                                    # 
                                    tt[j,0:6] = gpst[0:6]                      # YMDHM (no seconds).
                                    tt[j,5] = gpst[5]+gpst[6]/1e3+gpst[7]/1e6  
                                    gps[j,0] = gpst[8]                         # SRT: Latitude.
                                    gps[j,1] = gpst[9]                         # SRT: longitude.
                                    gps[j,9] = gpst[10]                        # SRT: similar to height above take off.
                                    gps[j,14] = poly(gpst[10])/1000;           # Above takeoff height derived scale.
                                    #gps[j,14] = (cc[0]*gpst[10] + cc[1])/1e3; 
                                        
                    # Extra correction for drone photos and drone video photos.
                    if len(fct)>0 and isdrone:
                        # For drone photos.
                        if len(picstr) == 2:                                       
                            tj = etime(fct[1][0,1:7],tt[j,:])
                            if tj>=fct[1][0,0] and  tj<=fct[1][-1,0]:
                                tx = fct[1][:,0]
                                h4 = fct[1][:,[13,10,9,23]]                    # 4 heights from drone flight record.
                                zz = fct[1][:,11]                              # ground_elevation_at_drone_location.
                                h4p = interp1(tx,h4,tj,method='linear')        # Interpolated heights for photos.
                                scp = poly(h4p)/1000
                                zzp = interp1(tx,zz,tj,method='linear')
                                gps[j,[7,8,10]] = h4p[0,[0,1,3]]               # Update heights.
                                gps[j,[12,13,15]] = scp[0,[0,1,3]]             # Update scales.
                                gps[j,2] = zzp[0]                              # % Overwrite photo elevation if drone elevation found.
                                if gps[j,9] == zmax:                           # Only updating height without values.
                                    gps[j,9] = h4p[0,2]                        # Updating height.          
                                    gps[j,14] = scp[0,2]                       # Updating scale.
 
                                # Identifying speed from fct if availabe.
                                logtargets = ['xSpeed(mps)','ySpeed(mps)','zSpeed(mps)']
                                idx = lognames.index(logtargets[0])
                                idy = lognames.index(logtargets[1])
                                idz = lognames.index(logtargets[2])
                                uxyz = fct[1][:,[idx,idy,idz]]
                                uxyzp = interp1(tx,uxyz,tj,method='linear')    # Interpolated velocity.
                                ux,uy,uz = uxyzp[0]
                                uxy = np.sqrt(ux**2 + uy**2)                   # Average horizontal speed.    
                                
                        # For drone video derived photos.
                        elif len(picstr) == 3:
                            tj = etime(fct[0][0,1:7],tt[j,:])
                            if tj>=fct[0][0,0] and  tj<=fct[0][-1,0]:
                                tx = fct[0][:,0]
                                h4 = fct[0][:,[13,10,9,23]]                    # 4 heights from drone flight record.
                                zz = fct[0][:,11]                              # ground_elevation_at_drone_location.
                                h4p = interp1(tx,h4,tj,method='linear')        # Interpolated heights for photos.
                                scp = poly(h4p)/1000
                                #scp = (cc[0]*h4p+cc[1])/1000.0                # Update scales.
                                zzp = interp1(tx,zz,tj,method='linear')
                                gps[j,[7,8,10]] = h4p[0,[0,1,3]]               # Update heights.
                                gps[j,[12,13,15]] = scp[0,[0,1,3]]             # Update scales.
                                gps[j,2] = zzp[0]                              # % Overwrite photo elevation if drone elevation found.
                                if gps[j,9] == zmax:                           # Only updating height without values.
                                    gps[j,9] = h4p[0,2]                        # Updating height.          
                                    gps[j,14] = scp[0,2]                       # Updating scale.
                                    
                                # Identifying speed from fct if availabe.
                                logtargets = ['xSpeed(mps)','ySpeed(mps)','zSpeed(mps)']
                                idx = lognames.index(logtargets[0])
                                idy = lognames.index(logtargets[1])
                                idz = lognames.index(logtargets[2])
                                uxyz = fct[0][:,[idx,idy,idz]]
                                uxyzp = interp1(tx,uxyz,tj,method='linear')    # Interpolated velocity.
                                ux,uy,uz = uxyzp[0]
                                uxy = np.sqrt(ux**2 + uy**2)                   # Average horizontal speed.
                                
                    tsec =  tt[j,5]
                    y, m, d, H, M = tt[j,0:5].astype(int)
                    if tsec<0:
                        dtm_new = datetime(y, m, d, H, M, 0)
                    else:
                        dtm_new = datetime(y, m, d, H, M, int(tsec))
                    dtm_str = dtm_new.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Extracting photo resolution information.
                    [x11,y11,x12,y12,x21,y21,x22,y22] = [zmax]*8               # Default values.
                    [sc1,sc2,res1,res2, ww,hh,err] = [zmax]*7
                    [res12, res12m] = [-1]*2                                   # Set default resolution -1 m/pixel.
                    scale_source = 'N/A'

                    # Updating data if included in SCTable.
                    if len(SCTable)>0 and pic_stem in SCTable['Name_stem'].values:
                        rj = SCTable.loc[SCTable['Name_stem'] == pic_stem]
                        
                        # Further check if the two image, to avoid identical  #
                        # image name of photos from two cameras.
                        ta = pd.to_datetime(rj.Date_PST.iloc[0], errors='coerce')
                        tb = pd.to_datetime(dtm_str, errors='coerce')
                        dts = abs((ta - tb).total_seconds())
                        dist = gpsdist(rj.Latitude.iloc[0],rj.Longitude.iloc[0],gps[j,0],gps[j,1])    
                        cond1 = keywords[j].lower() != 'img'
                        cond2 = (not cond1) and mak == rj.Make.iloc[0] and mdl == rj.Model.iloc[0]
                        cond3 = dts<=61                                        # 61 s.
                        cond4 = dist<=100                                      # 100 m.
                        cond5 = keywords[j].lower() == 'img' and len(picstr) == 3 and ispng
                        
                        if cond1 or cond2 or cond3 or cond4 or cond5:
                            res12 = rj['L1L2_res_meter_per_pixel'].iloc[0]
                            if res12>0:                                        # Update data only when res12 is positive.
                                scale_source = rj['Scale_source'].iloc[0]
                                x11 = rj['L1_px1_pixel'].iloc[0]
                                y11 = rj['L1_py1_pixel'].iloc[0]
                                x12 = rj['L1_px2_pixel'].iloc[0]
                                y12 = rj['L1_py2_pixel'].iloc[0]
                                x21 = rj['L2_px1_pixel'].iloc[0]
                                y21 = rj['L2_py1_pixel'].iloc[0]
                                x22 = rj['L2_px2_pixel'].iloc[0]
                                y22 = rj['L2_py2_pixel'].iloc[0]
                                sc1 = rj['L1_marker_meter'].iloc[0]
                                sc2 = rj['L2_marker_meter'].iloc[0]
                                res1 = rj['L1_res_meter_per_pixel'].iloc[0]
                                res2 = rj['L2_res_meter_per_pixel'].iloc[0]
                                res12 = rj['L1L2_res_meter_per_pixel'].iloc[0]
                                res12m = rj['L1L2_res_millimeter_per_pixel'].iloc[0]
                                ww = res12*w
                                hh = res12*h 
                                uxy0 = rj['Speed_meter_per_second'].iloc[0]
                                ux0 = rj['Speed_x_meter_per_second'].iloc[0]
                                uy0 = rj['Speed_y_meter_per_second'].iloc[0]
                                uz0 = rj['Speed_z_meter_per_second'].iloc[0]
                                fr0 = rj['Frame_rate_frames_per_second'].iloc[0]
                                dr0 = rj['Duration_second'].iloc[0]
                                if uxy0 != zmax: uxy = uxy0
                                if ux0 != zmax: ux = ux0
                                if uy0 != zmax: uy = uy0
                                if uz0 != zmax: uz = uz0
                                if fr0 != zmax: fr = fr0
                                if dr0 != zmax: dr = fr0
                                err = rj['Relative_error_percent'].iloc[0]
                                gps[j,0] = rj['Latitude'].iloc[0]
                                gps[j,1] = rj['Longitude'].iloc[0]
                                gps[j,2] = rj['Altitude_meter'].iloc[0]
                                gps[j,4] = rj['Latitude_SFA'].iloc[0]
                                gps[j,5] = rj['Longitude_SFA'].iloc[0]
                                rjdate = pd.to_datetime(rj['Date_PST'], errors='coerce')
                                if rjdate.iloc[0] != refdate:
                                    dtm_new = pd.to_datetime(rj['Date_PST'].iloc[0], errors='coerce')
                                    tsec = rj['Date_Second'].iloc[0]
                                    if tsec>0: 
                                        dtm_new = dtm_new.floor('min') + \
                                        pd.to_timedelta(tsec, unit='s')
                                    dtm_str = dtm_new.strftime("%Y-%m-%d %H:%M:%S")
                                if rj['Site_ID'].iloc[0] != 'N/A': sites[j] = rj['Site_ID'].iloc[0] 
                                if rj['Video_ID'].iloc[0] != 'N/A': vidx = rj['Video_ID'].iloc[0] 
                                if rj['Notes'].iloc[0] != 'N/A': note = rj['Notes'].iloc[0]
                                if rj['Make'].iloc[0] != 'N/A': mak = rj['Make'].iloc[0]
                                if rj['Model'].iloc[0] != 'N/A': mdl = rj['Model'].iloc[0]
                                if rj['Author'].iloc[0] != 'N/A': author = rj['Author'].iloc[0]
                                
                    # For video-derived images with user video metadata.
                    if len(SCTable)>0 and len(picstr)==3 and \
                    '_'.join(picstr[0:2]) in SCTable['Name_stem'].values:
                        v2p_stem = '_'.join(picstr[0:2])
                        rjj = SCTable.loc[SCTable['Name_stem'] == v2p_stem]
                        if gps[j,0] == zmax or gps[j,1] == zmax: 
                            gps[j,0] = rjj['Latitude'].iloc[0]
                            gps[j,1] = rjj['Longitude'].iloc[0]
                            gps[j,2] = rjj['Altitude_meter'].iloc[0]
                        if dtm_new == refdate: 
                            dtm_new = pd.to_datetime(rjj['Date_PST'].iloc[0], errors='coerce')
                            tsec = rjj['Date_Second'].iloc[0]
                            if tsec>0: dtm_new += pd.to_timedelta(tsec, unit='s')
                            dtm_str = dtm_new.strftime("%Y-%m-%d %H:%M:%S")
                            
                        if sites[j] == 'N/A':sites[j] = rjj['Site_ID'].iloc[0]
                        if vidx == 'N/A': vidx = rjj['Video_ID'].iloc[0]
                        if note == 'N/A': note = rjj['Notes'].iloc[0]
                        if mak == 'N/A': mak = rjj['Make'].iloc[0]
                        if mdl == 'N/A': mdl = rjj['Model'].iloc[0]
                        if author == 'N/A': author = rjj['Author'].iloc[0]
                        if uxy == zmax: uxy = rjj['Speed_meter_per_second'].iloc[0]
                        if ux == zmax: ux = rjj['Speed_x_meter_per_second'].iloc[0]
                        if uy == zmax: uy = rjj['Speed_y_meter_per_second'].iloc[0]
                        if uz == zmax: uz = rjj['Speed_z_meter_per_second'].iloc[0]
                        if fr == zmax: fr = rjj['Frame_rate_frames_per_second'].iloc[0]
                        if dr == zmax: dr = rjj['Duration_second'].iloc[0]
                        
                        # Do not update time in the following cases:
                        # a) both fligth log and SRT exist;
                        # b) SRT exsits. 
                        if fr >0 and vid_stem not in srts:
                            fmid = float(pic_stem.split("_", 2)[2])
                            tsecnew = fmid*1/fr
                            tsec = tsecnew if tsec <0 else tsec + tsecnew
                            dtm_new = pd.to_datetime(dtm_new)
                            dtm_new = dtm_new.floor('min') + pd.to_timedelta(tsec, unit='s')
                            tsec = dtm_new.second + dtm_new.microsecond/1e6 + dtm_new.nanosecond/1e9
                            dtm_new = dtm_new.floor('s')
                            dtm_str = dtm_new.strftime("%Y-%m-%d %H:%M:%S")
                            
                    # Updating data if photo resolution is not found.
                    if scsource in 'ai' and res12<0:
                        # Define image size used for AI inference.
                        img_type = ['default','image']
                        img_type_err = ', '.join(img_type)
                        if PP.ImageSizeType.lower() == 'default':
                            img_size0 = int(int(model_name.split('.')[1])*scalefactor)
                            img_size = (img_size0,img_size0)
                        elif PP.ImageSizeType.lower() == 'image':
                            img_size = (int(w*scalefactor),int(h*scalefactor))
                        else:
                            msg = f"Invalid image type: {PP.ImageSizeType}." + \
                                f"Options: {img_type_err}"
                            raise ValueError(msg)
                        if printimgsize: print(f'Inference image size: {img_size}')
                        
                        # Generate parameters used for AI inference.
                        max_det = PP.MaximumNumberOfObject
                        data = {"imgsz": img_size, "conf": conf1, "iou": iou,"max_det":max_det}
                        
                        # Access AI from cloud.
                        names = []; values  = []
                        if cloud_name.lower() == 'ultralytics':
                            headers = {"x-api-key": api_key}
                            with open(pic_path, "rb") as f:
                                response = requests.post(model_id, headers=headers, data=data, files={"file": f})
                                
                            # Moving to the next predictin if status_code is not 200.
                            status = response.status_code
                            if status != 200: continue
                            results = response.json()['images'][0]['results']
                            
                            # Converting the results to array.
                            names = []; values  = []
                            for det in results:
                                x1 = round(det['box']['x1'])
                                y1 = round(det['box']['y1'])
                                x2 = round(det['box']['x2'])
                                y2 = round(det['box']['y2'])
                                cls_idx = det['class']
                                cls_name = det['name']
                                conf = det['confidence']
                                names.append(cls_name)
                                values.append([x1, y1, x2, y2, cls_idx, conf])
                            names = np.array(names, dtype=str)            
                            values  = np.array(values, dtype=pr)
                            
                        elif cloud_name.lower() == '' and cmpt in 'local':     # Access AI from local pre-trained models.
                            results = model.predict(
                                source=str(pic_path),                          # can be path, OpenCV array, folder, glob, URL, webcam…
                                **data,
                                project= DirectoryL1,
                                save= PP.YOLOSave,                             # save annotated images to `./runs`
                                name = sdir,
                                exist_ok= True,                                # overwrite instead of incrementing
                                show=False,                                    # pop up a window with boxes
                                verbose=PP.YOLOPrint,                          # console message control.
                                line_width=1)                                  # box border thickness in pixel.
                            
                            # Data format conversion.
                            names = []; values  = []
                            for r in results:
                                for xyxy, conf, cls in zip(
                                        r.boxes.xyxy, r.boxes.conf, r.boxes.cls
                                        ):
                                    x1, y1, x2, y2 = map(float, xyxy)
                                    names.append(model.names[int(cls)])
                                    values.append([
                                        round(x1), 
                                        round(y1), 
                                        round(x2), 
                                        round(y2),
                                        int(cls),
                                        float(conf)])
                            names = np.array(names, dtype=str)
                            values  = np.array(values, dtype=pr)
                            if len(names)>0: status = 200
                            
                        # Coputing resolution and uncertainty.
                        if len(names) >0: 
                            pmin = 2                                           # Minimum pixel of sediment away from the photo boundary.
                            tfs = ((values[:, 0] > pmin)  & (values[:, 1] > pmin)  & 
                                (values[:, 2] < w - pmin)  & (values[:, 3] <= h - pmin) 
                                & (values[:,5]>=conf1))
                            names = names[tfs]; values = values[tfs,:]
                            nrc = len(names)
                            scname = '_'.join(list(set(names[:nrc])))
                            if nrc == 1:
                                scs = np.array([ sc[name] for name in names ])
                                [x11,y11,x12,y12] = values[0,0:4]
                                sc1 = scs[0]
                                res1_w = sc1/(x12 - x11)
                                res1_h =  sc1/(y12 - y11)           
                                res1 = (res1_w + res1_h)/2.0
                                res12 = res1
                                res12m = res1*1000.0
                                ww = w*res12; hh = h*res12
                                err = (res1_w - res12)/res12*100
                                scale_source = scname
                            elif nrc>1:
                                scs = np.array([ sc[name] for name in names ])
                                [x11,y11,x12,y12] = values[0,0:4]
                                [x21,y21,x22,y22] = values[1,0:4]
                                sc1 = scs[0]
                                sc2 = scs[1]
                                res1_w = sc1/(x12 - x11)
                                res1_h =  sc1/(y12 - y11)   
                                res2_w = sc2/(x22 - x21)
                                res2_h =  sc2/(y22 - y21)   
                                res1 = (res1_w + res1_h)/2.0
                                res2 = (res2_w + res2_h)/2.0
                                res12 = (res1 + res2)/2.0
                                res12m = res12 * 1000.0
                                ww = w*res12; hh = h*res12
                                err = (res1 - res12)/res12*100                 # Uncertainty of reference scale.
                                scale_source = scname
                    elif scsource in 'data' and res12<0:
                        clspath = os.path.join(DirectoryL1,mdir,'classes.txt')
                        txtpath = os.path.join(DirectoryL1,mdir,pic_stem + '.txt')
                        if os.path.isfile(clspath) and os.path.isfile(txtpath) \
                            and os.path.getsize(txtpath)>0:
                            clss = pd.read_csv(clspath, header=None)[0].tolist()                             
                            scs = {k: sc[k] for k in clss if k in sc}
                            pair = {s:scs[c] for s, c in enumerate(clss) if c in scs}
                            td = np.loadtxt(txtpath)
                            if len(td.shape) == 1: td = td.reshape(1,-1)
                            scn = np.array([pair.get(int(idx), np.nan) for idx in td[:, 0]])
                            td = td[~np.isnan(scn),:]
                            scn = scn[~np.isnan(scn)]
                            names = np.array([clss[int(idx)] if int(idx) < len(clss) \
                                              else "unknown" for idx in td[:, 0]])
                            nrc = td.shape[0]
                            scname = '_'.join(list(set(names[:nrc])))
                            if nrc == 1:
                                x11 = round((td[0,1] - td[0,3]/2)*w)
                                y11 = round((td[0,2] - td[0,4]/2)*h)
                                x12 = round((td[0,1] + td[0,3]/2)*w)
                                y12 = round((td[0,2] + td[0,4]/2)*h)
                                sc1 = scn[0]
                                res1_w = sc1/(x12 - x11)
                                res1_h =  sc1/(y12 - y11)           
                                res1 = (res1_w + res1_h)/2.0
                                res12 = res1
                                res12m = res1*1000.0
                                ww = w*res12; hh = h*res12
                                err = (res1_w - res12)/res12*100
                                scale_source = scname
                            if nrc == 2:
                                x11 = round((td[0,1] - td[0,3]/2)*w)
                                y11 = round((td[0,2] - td[0,4]/2)*h)
                                x12 = round((td[0,1] + td[0,3]/2)*w)
                                y12 = round((td[0,2] + td[0,4]/2)*h)
                                x21 = round((td[1,1] - td[1,3]/2)*w)
                                y21 = round((td[1,2] - td[1,4]/2)*h)
                                x22 = round((td[1,1] + td[1,3]/2)*w)
                                y22 = round((td[1,2] + td[1,4]/2)*h)  
                                sc1 = scn[0]
                                sc2 = scn[1]
                                res1_w = sc1/(x12 - x11)
                                res1_h =  sc1/(y12 - y11)   
                                res2_w = sc2/(x22 - x21)
                                res2_h =  sc2/(y22 - y21)   
                                res1 = (res1_w + res1_h)/2.0
                                res2 = (res2_w + res2_h)/2.0
                                res12 = (res1 + res2)/2.0
                                res12m = res12 * 1000.0
                                ww = w*res12; hh = h*res12
                                err = (res1 - res12)/res12*100                 # Uncertainty of reference scale.
                                scale_source = scname
                    elif len(fct)>0 and isdrone and gps[j,12] != zmax and res12<0:
                        res12 = gps[j,12]
                        res12m = res12 * 1000.0
                        ww = w*res12
                        hh = h*res12
                        scale_source = 'log'
                    ww = abs(res12) * w
                    hh = abs(res12) * h 
                    
                    
                    # Combining all data in a single list.
                    if isinstance(vidx, (int, float, np.number)): vidx = str(int(vidx))
                    update_list = {
                        "Name": pic_name,
                        "Folder": one_dir,
                        "Scale_source": scale_source,
                        "L1_px1_pixel": x11,
                        "L1_py1_pixel": y11,
                        "L1_px2_pixel": x12,	
                        "L1_py2_pixel": y12,	
                        "L2_px1_pixel": x21,	
                        "L2_py1_pixel": y21,	
                        "L2_px2_pixel": x22,	
                        "L2_py2_pixel": y22,	
                        "L1_marker_meter": sc1,	
                        "L2_marker_meter": sc2,	                    
                        "Width_pixel": w,
                        "Height_pixel": h,
                        "L1_res_meter_per_pixel": res1,	
                        "L2_res_meter_per_pixel": res2,
                        "L1L2_res_meter_per_pixel": res12,	
                        "L1L2_res_millimeter_per_pixel": res12m,
                        "Scale_x_meter": ww,	
                        "Scale_y_meter": hh,
                        "Relative_error_percent": err,   
                        "Valid": 1,
                        "Corrected": 0,
                        "Usage": usage,
                        "Photo_indicator": pv_indiactor,                       # 0 for video, 1 for photo, 2 for video frame.
                        "Latitude": gps[j,0],                                  # From photo exif, video exif, or log.
                        "Longitude": gps[j,1],                                 # From photo exif, video exif, or log.
                        "Latitude_SFA": gps[j,4],                              # From file.
                        "Longitude_SFA": gps[j,5],                             # From file.
                        "Altitude_meter": gps[j,2],                            # From photo exif, video exif, or log.
                        "Altitude_reference": gps[j,3],                        # From photo exif, video exif, or log.
                        "Height_m": gps[j,7],                                  # Sonar height.
                        "Height_manual_m": gps[j,6],                           # Default manual height.
                        "Height_sonar_m": gps[j,7],                            # Sonar height.
                        "Height_above_drone_m": gps[j,8],                      # Above drone height.
                        "Height_above_takeoff_m": gps[j,9],                    # Above take off height.
                        "Height_velocity_integrate_m": gps[j,10],              # Velocity integration height.
                        "Resolution_manual_meter_per_pixel": gps[j,11],
                        "Resolution_sonar_meter_per_pixel": gps[j,12],
                        "Resolution_above_drone_meter_per_pixel": gps[j,13],
                        "Resolution_above_takeoff_meter_per_pixel": gps[j,14],	
                        "Resolution_velocity_integrate_meter_per_pixel": gps[j,15],
                        'Speed_meter_per_second': uxy,
                        'Speed_x_meter_per_second': ux,
                        'Speed_y_meter_per_second': uy,
                        'Speed_z_meter_per_second': uz,
                        'Frame_rate_frames_per_second': fr,
                        'Duration_second': dr,
                        "Date_PST": dtm_str,
                        "Date_Second": tsec,
                        "Make": mak,
                        "Model": mdl,
                        "Author": author,
                        "Keyword": keywords[j],
                        "Site_ID": sites[j] if stid == 'N/A' else stid,
                        'Video_ID': vidx,
                        "Notes": note
                    }
                    defaults_dict_new = defaults_dict.copy()
                    defaults_dict_new.update(update_list)
                    record_list.append(defaults_dict_new)
                else:
                    res12m = rowj.L1L2_res_millimeter_per_pixel.values[0]
                    dtm = rowj.Date_PST.values[0]
                    dtm_str = pd.to_datetime(dtm)
                    tsec = rowj.Date_Second.values[0]
                    scale_source = rowj.Scale_source.values[0]
                    defaults_dict_new = rowj.iloc[0].to_dict()
                    update_list = {
                        "Folder": one_dir,
                        "Date_PST": dtm_str,
                        "Date_Second": tsec,
                        "Site_ID": sites[j] if stid == 'N/A' else stid,
                        'Video_ID': vidx,
                        "Notes": note
                        }
                    defaults_dict_new.update(update_list)
                    record_list.append(defaults_dict_new)
                    
                t_end = time.time()
                dt = t_end - t_start
                if res12m == -1:
                    print(formatSpec2a %(count,nvt,j+1,npics,pic_name,scale_source,\
                                        abs(res12m), status, 100*(j+1)/npics, dt))
                    if scsource in 'data' and not os.path.isfile(clspath):
                        print(f'Warning: manual scale data not found in {clspath}...')
                else:
                    print(formatSpec2 %(count,nvt,j+1,npics,pic_name,scale_source,\
                                        res12m, status, 100*(j+1)/npics, dt))
                
            # Convert the list to a DataFrame
            scales_p = pd.DataFrame(record_list, columns=cnames)
                                
        if scales_p.empty and not scales_v.empty: scales_df = scales_v.copy()
        if scales_v.empty and not scales_p.empty: scales_df = scales_p.copy()
        if (not scales_v.empty) and (not scales_p.empty): 
            scales_df = pd.concat([scales_p, scales_v], axis=0, ignore_index=True)
            
        # Save the final CSV
        if os.path.isfile(snamexls):
            os.remove(snamexls)
        if len(scales_df)>0: 
            scales_df['Video_ID'] = scales_df['Video_ID'].astype(str).str.strip()
            scales_df.to_csv(snamexls, index=False)
        #tts = time.time() - t0
        #print(formatSpec3 %(tts,tts/nvt)) if nvt>0 else print(formatSpec4 %(tts))
        
    return scales_df 

#%% Generating metadata for L2 folders where it contains multiple L1 folders.
def metadataLevel2(PP):
    
    # Checking if working diretory has been specified.
    if PP.Directory == '': raise ValueError('Working directory is not defined.')
    
    # Pass parameters from PP to local.
    DirectoryL2 = PP.Directory   
    SCName= PP.ScaleTemplateName
    OverWriteSC = PP.OverWriteScaleFile
    OverWriteAll = PP.OverWriteAll
    CSVEncoding = PP.CSVEncoding
    default_excludes = {".", ".."}
    user_excludes = PP.ExcludingFolders
    
    # Setups.
    [SCNameStem,ext] = SCName.split('.')
    snamexls_all = DirectoryL2 + os.sep + SCNameStem + '_' + \
        os.path.basename(DirectoryL2) + '.' + ext
    #pd.set_option('future.no_silent_downcasting', True)
    scales_all = pd.DataFrame()
    
    if os.path.isfile(snamexls_all) and (not OverWriteSC) and (not OverWriteAll):
        print("Scale file existed, skip.")
        scales_all = pd.read_csv(snamexls_all, encoding= CSVEncoding,keep_default_na=False)
    else:
        # Defining folders to be exluded for processing.
        if isinstance(user_excludes, str):
            user_excludes = {user_excludes.lower()}
        else:
            user_excludes = {u.lower() for u in user_excludes}
        all_excludes = {d.lower() for d in default_excludes} | user_excludes
        
        folders = [
            name for name in os.listdir(DirectoryL2)
            if os.path.isdir(os.path.join(DirectoryL2, name))
               and name.lower() not in all_excludes]
        folders = natsorted(folders)
        
        scales_list = []
        PP1 = copy.deepcopy(PP)
        for i in range(len(folders)):
            DirectoryL1 = DirectoryL2 + os.sep + folders[i]
            PP1.Directory = DirectoryL1
            scales = metadataLevel1(PP1)
            if not scales.empty:
                scales_list.append(scales)
            print()
        
        if len(scales_list)>0:
            scales_all = pd.concat(scales_list, ignore_index=True)
            if PP.MetadataUnique:
                scales_all = scales_all.drop_duplicates(subset=['Name'],keep='first')
                
            if os.path.isfile(snamexls_all):
                os.remove(snamexls_all)
            scales_all.to_csv(snamexls_all, index=False)
        
    return scales_all

#%% Generating metadata based on given parameters.
def metadata(PP):
    formatSpec0 = 65*'='
    formatSpec1 = 'Total time: %3.2f s, time per photo %3.2f s'
    formatSpec2 = 'Total time: %3.2f s, time per video %3.2f s'
    formatSpec3 = 'Total time: %3.2f s, time per photo/video %3.2f s'
    formatSpec4 = 'Total time: %3.2f s, no scales generated.'
    formatSpec5 = 65*'-' +'\n' 
    
    print(formatSpec0)
    t_start  = time.time()
    if PP.FolderLevel.lower() == 'l1':
        scales =  metadataLevel1(PP)
    elif PP.FolderLevel.lower() == 'l2':
        scales =  metadataLevel2(PP)
    else:
        raise 'Invalid metadata level option'
        
    nf = scales.shape[0]
    t_end = time.time()
    tt = t_end - t_start
    if nf == 0:
        print(formatSpec4 %(tt))
    else:
        if np.all(scales.Photo_indicator.to_numpy() > 0):
            print(formatSpec1 %(tt, tt/nf))
        elif np.all(scales.Photo_indicator.to_numpy() == 0):
            print(formatSpec2 %(tt, tt/nf))
        else:
            print(formatSpec3 %(tt, tt/nf))
    print(formatSpec5)
    
    return scales

#%% The default data structure from AI models.                                #
def defaultResults(PP):
    
    cloud_name, api_key, model_id = cloud(PP)
    if cloud_name.lower() == 'ultralytics':
        results = [
            {
                'box': {'x1': 0, 'x2': 1, 'y1': 0, 'y2': 1}, 
                'class': 0, 'confidence': 1,'name': 'rock'},
            {
                'box': {'x1': 0, 'x2': 2, 'y1': 0, 'y2': 2}, 
                'class': 0,'confidence': 1, 'name': 'rock'}
            ]
    elif cloud_name.lower() == 'landingai':
        if PP.LandingAIInferenceMethod.lower() == 'landingai' and platform.system() == 'Windows':
            results = [
                detectionResult(label_index=1,bboxes=[0, 0, 1, 1],score=1,label_name='rock'),
                detectionResult(label_index=1,bboxes=[0, 0, 2, 2],score=1,label_name='rock')
                ]
        else:
            results = {
                "3e715588-7b17-4cbe-bfdc-a1fd9280baa2": 
                    {'score': 1, 'labelName': 'rock', 'labelIndex': 1,
                     'coordinates': {'xmin': 0, 'ymin': 0, 'xmax': 1, 'ymax': 1}, 
                     'defect_id': 139811}, 
                "05d9b034-1976-4e2d-bccd-184fcad07a84": 
                    {'score': 1, 'labelName': 'rock', 'labelIndex': 1, 
                     'coordinates': {'xmin': 0, 'ymin': 0, 'xmax': 2, 'ymax': 2}, 
                     'defect_id': 139811}
                    }
    else:
     results = [
         {
             'box': {'x1': 0, 'x2': 1, 'y1': 0, 'y2': 1}, 
             'class': 0, 'confidence': 1,'name': 'rock'},
         {
             'box': {'x1': 0, 'x2': 2, 'y1': 0, 'y2': 2}, 
             'class': 0,'confidence': 1, 'name': 'rock'}
         ]
     
    return results

#%% Converting txt files into YOLO format.
def txtformat(txtFolderPath, pattern):  
    formatSpec0 = 'Folder: %s'
    formatSpec1 = '%s (%g in %g, %.2f%%)'
    formatSpec2 = '%s is not converted due to zero size'
    
    folders = [f for f in os.listdir(txtFolderPath) \
               if os.path.isdir(os.path.join(txtFolderPath, f))]
        
    if len(folders) == 0:
        dirname = os.path.dirname(txtFolderPath)
        folders = [os.path.basename(txtFolderPath)]
    else:
        dirname = txtFolderPath
    
    for folder in folders:
        print(65*'-')
        print(formatSpec0 %(folder))
        folderpath = dirname + os.sep + folder
        file_pattern = os.path.join(folderpath, pattern)
        files = glob.glob(file_pattern)
        names = [os.path.basename(f) for f in files]
        nf = len(files)
        
        # iterate oer matched files
        for i in range(len(files)):
            print(formatSpec1 %(names[i], i+1, nf, (i+1)/nf*100))
            txt = np.loadtxt(files[i])                                         # Loading txt data.
            if txt.size>0:
                if txt.ndim == 1: txt = txt.reshape(1,-1)                      # Converting to 1 row if 1D.
                nv = txt.shape[1]
                fmti = ('%d',) + ('%8.6f',) * (nv - 1)
                np.savetxt(files[i], txt, fmt = fmti,delimiter=" ")
            else:
                print(formatSpec2 %(files[i]))
        print('')
        
#%% Compare the differences in two labels.                                    #
def labelCompare(PathA, PathB, tol = 1e-4, overwrite = True,\
                 fig_show = False, dpi_overlay = 1200):
    excluded_files = {"classes.txt", "*_p.txt"}
    
    # Output information control.
    formatSpec1 = '%s (%g in %g, %.2f%%)'
    #formatSpec2 = '%s is not converted due to zero size'
    valid_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}            # Extension of valid photos.
    
    # Obtaining image extensions of folders in path A and path B.
    ext1 = list({os.path.splitext(f)[1].lower() for f in os.listdir(PathA)
                    if os.path.splitext(f)[1].lower() in valid_exts})
    # ext2 = list({os.path.splitext(f)[1].lower() for f in os.listdir(PathB)
    #                 if os.path.splitext(f)[1].lower() in valid_exts})
    ext_final = ext1; Path_final = PathA
    
    # Obtaning YOLO label text file names in both folders.
    labels_a = [file for file in os.listdir(PathA)
                if file.endswith('.txt') and 
                not any(fnmatch.fnmatch(file, pattern) for pattern in excluded_files)]
    labels_b = [file for file in os.listdir(PathB)
                if file.endswith('.txt') and 
                not any(fnmatch.fnmatch(file, pattern) for pattern in excluded_files)]
    labels = list(set(labels_a) & set(labels_b))
    nf = len(labels)
    
    # Checking label differennces and visualize the difference
    for i in range(len(labels)):
        print(formatSpec1 %(labels[i], i+1, nf, (i+1)/nf*100))
        outFileName = Path_final + os.sep + labels[i].replace('.txt','_predict.jpg')
        outImageName = os.path.basename(outFileName)
        if os.path.isfile(outFileName) and not overwrite: 
            print(f'{outImageName} has been generated, skipping...')
            continue
        
        # Processing photos without difference checking.
        for ext in ext_final:
            imgpath = os.path.join(Path_final, labels[i].replace('.txt',ext))
            if not os.path.isfile(imgpath): continue            
            txt1 = np.loadtxt(PathA + os.sep + labels[i])                      # Loading txt data.
            txt2 = np.loadtxt(PathB + os.sep + labels[i])                      # Loading txt data.
            if txt1.size>0 and txt2.size>0:
                imgi = Image.open(imgpath)                                     # Loading image.
                sw = imgi.size[0]
                sh = imgi.size[1]
                
                if txt1.ndim == 1: txt1 = txt1.reshape(1,-1)                   # Converting to 1 row if 1D.
                if txt2.ndim == 1: txt2 = txt2.reshape(1,-1)                   # Converting to 1 row if 1D.
                r1 =  np.sqrt(txt1[:,3]**2 + txt1[:,4]**2)
                r2 =  np.sqrt(txt2[:,3]**2 + txt2[:,4]**2)
                geo1 = np.column_stack((txt1[:,1:3],r1))
                geo2 = np.column_stack((txt2[:,1:3],r2))
                tfx, locc = ismembertol(geo1, geo2, byrows=True, tol = tol)
                bboxes = center2corners(txt1[:,1:3],txt1[:,3:5])
                bboxes[:,[0,2]] = bboxes[:,[0,2]]*sw
                bboxes[:,[1,3]] = bboxes[:,[1,3]]*sh
                bboxesa = bboxes[tfx,:].astype(int)
                bboxesb = bboxes[~tfx,:].astype(int)
                
                # Plottings.
                # Color: green (0,255,0); blue (0,0,255); red (255,0,0); whilte (255,255,255); yellow (255,255,0)
                imgi_rgb = np.array(imgi)
                line_color1 = (0,255,0)
                line_color2 = (255,0,0)
                thickness = 2
                plt.figure(1)
                for bbox in bboxesb:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(imgi_rgb, (x1, y1), (x2, y2), line_color2, thickness)
                for bbox in bboxesa:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(imgi_rgb, (x1, y1), (x2, y2), line_color1, thickness)
                plt.imshow(imgi_rgb,aspect=1)
                plt.axis("off")
                plt.savefig(outFileName, bbox_inches='tight',pad_inches=0, dpi=dpi_overlay)
                plt.show() if fig_show else plt.close(1)
                plt.close('all')
            else:
                print(f'No data found in {labels[i]}')
                
#%%
def photoCorrectionSingle(input_folder,output_folder,filename):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    try:
        img = Image.open(input_path)
    except (OSError, IOError) as e:
        print(f"Could not open {filename}: {e}")

    exif_data = img.info.get('exif')

    if exif_data and filename.lower().endswith(('.jpg', '.jpeg', '.tiff')):
        exif_dict = piexif.load(exif_data)
        orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)
        
        if orientation != 1:
            img_corrected = ImageOps.exif_transpose(img)
            exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
            exif_bytes = piexif.dump(exif_dict)
            img_corrected.save(output_path, exif=exif_bytes, quality=100)
            print(f"Corrected orientation for: {filename}")
        else:
            # Directly copy file as-is if no correction needed
            shutil.copy2(input_path, output_path)
            print(f"No correction needed, copied directly: {filename}")
    else:
        # Directly copy file as-is if no correction needed
        shutil.copy2(input_path, output_path)
        print(f"No correction needed, copied directly: {filename}")
        
#%% Correcting photo orientation to be consistant with Windows photo viewer.  #
def photoCorrection(input_folder, outputflag='corrected'):
    photo_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    output_folder = input_folder + os.sep + outputflag
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(photo_exts):
            continue 
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            img = Image.open(input_path)
        except (OSError, IOError) as e:
            print(f"Could not open {filename}: {e}")
            continue

        exif_data = img.info.get('exif')

        if exif_data and filename.lower().endswith(('.jpg', '.jpeg', '.tiff')):
            exif_dict = piexif.load(exif_data)
            orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)
            
            if orientation != 1:
                img_corrected = ImageOps.exif_transpose(img)
                exif_dict['0th'][piexif.ImageIFD.Orientation] = 1
                exif_bytes = piexif.dump(exif_dict)
                img_corrected.save(output_path, exif=exif_bytes, quality=100)
                print(f"Corrected orientation for: {filename}")
            else:
                # Directly copy file as-is if no correction needed
                shutil.copy2(input_path, output_path)
                print(f"No correction needed, copied directly: {filename}")
        else:
            # Directly copy file as-is if no correction needed
            shutil.copy2(input_path, output_path)
            print(f"No correction needed, copied directly: {filename}")

#%% Compress photos to target size.                                           #
def photoCompress(img_path,target_size,quality_step=1,print_info=False,\
                  save_folder='compressed'):
    
    formatSpec0 = 'Iteration %d, quality %d, file size %.2f MB'
    formatSpec1 = 'Found compressed file, skipping...'
    psz = os.path.getsize(img_path)/1024**2                                    # Initial file size.
    quality = 100                                                              # Initial quality.
    step = 0                                                                   # Step index.
    
    outdir = os.path.dirname(img_path) + os.sep + save_folder
    img_path_c = outdir + os.sep + os.path.basename(img_path)
    if os.path.isfile(img_path_c):
        if print_info: print(formatSpec1)
    else:
        if print_info: print(formatSpec0 % (step, quality, psz))               # Print information.
        img_path_c = img_path
        if psz > target_size:
            img = cv2.imread(img_path)                                         # Read the image.
            while psz > target_size:
                quality -= quality_step
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]        # Quality: 0 - 100.
                success, encoded_img = cv2.imencode('.jpg', img, encode_param)
                step += 1
                if success:
                    psz = encoded_img.size/1024**2
                    if print_info: print(formatSpec0 % (step, quality, psz))       # Print information.
            if success: 
                
                os.makedirs(outdir, exist_ok=True)
                decoded = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                cv2.imwrite(img_path_c,decoded,[cv2.IMWRITE_JPEG_QUALITY, quality])
        
    return img_path_c

#%% Identifying non-sediment background using segmentation AI.
def backgroundLevel1(PP):
    
    # Printing information.
    formatSpec0 = 65*'-' 
    formatSpec1 = 'Folder name: %s'
    formatSpec2 = 'output name: %s'   
    formatSpec3 = 'Model name: %s from %s for %s'
    formatSpec5 = 'Model layer: %s, model parmeter: %4.2fM, submodule number: %s' 
    formatSpec6 = 'IOU: %3.2f, user confidence: %3.2f'
    formatSpec7 = 'Number of photos: %s, number of exluded photos %s'
    formatSpec8 = '%s: %s/%s, %s, mask existed, skipping... (%3.2f%%)'
    formatSpec9 = '%s: %s/%s, %s, no objects (%3.2f%%)'
    formatSpec10 = '%s: %s/%s: %s (%3.2f%%)'
    formatSpec11 = 'Total time: %3.2f s, time per photo %3.2f s'
    # #formatSpec3 = 'Segmentating non-sediment using model %s in %s'

    print(formatSpec0)
    t_start  = time.time()
    
    # Transfer parameters.
    model_name = PP.ModelName
    WorkDir = PP.Directory
    pext = PP.TargetPhotoExtension
    obj_name = PP.ObjectName
    conf1 = PP.ConfidenceThresholdUser                                         # User confidence threshold.
    iou = PP.YOLOIOU   
    exludes = PP.ExcludingPhotos
    outkeyword = PP.OutputKeyWord
    validationkeyword = PP.ValidationPlotKeyWord
    OverWriteSeg = PP.OverWriteSegementation
    OverWrtieAll = PP.OverWriteAll
    scalefactor = PP.ImageSizeFactor                                           # Used to control image size for inference.
    printimgsize = PP.PrintImageSize
    classes = PP.SegmentationClassName
    
    # Where to save the segmented images.
    foldername_save = PP.SegmentationSaveFolderName
    SaveDir = WorkDir + os.sep + foldername_save
    cname = os.path.basename(WorkDir)
    
    # Loading AI model.
    cloud_models=['YOLO11m.640.20250527','YOLO11m.1280.20250906','YOLO11m.640.20250906']
    cmpt = PP.ComputingSource.lower()
    if any(model_name.lower() == item.lower() for item in cloud_models):
        cloud_name, api_key, model_id = agent(PP)
        
        # Loading YOLO model
        if cmpt in 'local': 
            from ultralytics import YOLO
            model = YOLO(str(Path(model_id)))
            if classes is not None:
                classes = [i for i, c in model.names.items() if c in classes]
                
            import torch.nn as nn
            modules = list(model.model.modules())
            nmodules = len(modules)
            npp = sum(p.numel() for p in model.model.parameters())/1e6
            nl = sum(isinstance(m, nn.Conv2d) for m in modules)
            
    print(formatSpec1 %(cname))
    print(formatSpec2 %(foldername_save))
    print(formatSpec3 %(model_name, cmpt, obj_name))
    if cmpt in 'local': print(formatSpec5 %(nl, npp,nmodules))
    print(formatSpec6 %(iou,conf1))
    
    # Identifying photos for predictions. 
    files,stems,ext,ispv = getFiles(WorkDir,pext=pext)
    files = [files[i] for i in range(len(ispv)) if ispv[i]==0 and \
              outkeyword not in files[i] and validationkeyword not in files[i]]
    nft = len(files)
    files = [item for item in files if \
              not any(fnmatch.fnmatch(item, pattern) for pattern in exludes)]  # Exluding photos based on user inputs.
    files = natsorted(files)
    nf = len(files)
    print(formatSpec7 %(str(nf), str(nft - nf)))
    
    # Processing for each photo.
    for i in range(len(files)):
        fname = files[i]                                
        #ext = fname.split('.')[-1]
        stem = '.'.join(fname.split('.')[0:-1])
        segname = stem + '.tif'
        
        # Define output file names.
        outMask = SaveDir + os.sep + segname
        imgpath = WorkDir + os.sep + fname

        if os.path.isfile(outMask) and (not OverWriteSeg) and not OverWrtieAll:
            print(formatSpec8 %(cname, str(i+1), str(nf), fname,(i+1)/nf*100))
        else:
            imgi = Image.open(imgpath)
            sw = imgi.size[0]                                                  # True width of image (metadata).
            sh = imgi.size[1]                                                  # True height of image (metadata).
            
            # Define image size used for AI inference.
            img_type = ['default','image']
            img_type_err = ', '.join(img_type)
            if PP.ImageSizeType.lower() == 'default' or cmpt in 'cloud':       # img_size has to be single value for cloud inference.
                img_size0 = int(int(model_name.split('.')[1])*scalefactor)     # Single value 1280, becomes (1280x1280).
                img_size = (img_size0,img_size0)
            elif PP.ImageSizeType.lower() == 'image':
                img_size = (int(sw*scalefactor),int(sh*scalefactor))           # Width, height.
            else:
                msg = f"Invalid image type: {PP.ImageSizeType}." + \
                    f"Options: {img_type_err}"
                raise ValueError(msg)
            if printimgsize: print(f'Inference image size: {img_size}')
            
            # Preparing parameters for AI inference.
            max_det = PP.MaximumNumberOfObject
            data = {"imgsz": img_size, "conf": conf1, "iou": iou, "max_det": max_det,\
                    "device": None, "augment": False, "half": False, \
                    "classes": classes, "agnostic_nms": False, "save_crop": False, \
                    "save_json": False,  "save_txt": False, "retina_masks": True}
                
            # Calling AI models.
            if cmpt == 'cloud':
                if cloud_name.lower() == 'ultralytics':
                    headers = {"x-api-key": api_key}
                    data.update({'model': model_id})
                    url = "https://predict.ultralytics.com"
                    with open(imgpath, "rb") as f:
                        response = requests.post(url, headers=headers, data=data, files={"file": f})
                #     # Moving to the next predictin if status_code is not 200.
                #     if response.status_code != 200: continue
                #     results_ai = response.json()['images'][0]['results']
                    
                #     # Creating a fake data when number of objects less than 2.
                #     results=results0 if len(results_ai)<=2 else results_ai
                    
                #     #obj_name = results[0]['name']
                #     nrc = len(results)
                #     obj_id = []; obj_px = []; obj_score = []
                #     for k in range(nrc):
                #         obj_id.append(results[k]['class'])
                #         bb = [round(results[k]['box']['x1']),round(results[k]['box']['y1']),\
                #               round(results[k]['box']['x2']),round(results[k]['box']['y2'])]
                #         obj_px.append(bb)
                #         obj_score.append(results[k]['confidence'])
                # else:
                #     raise 'Invalid cloud name'
                
            elif cmpt == 'local':
                results = model.predict(
                    source=str(imgpath),                                       # can be path, OpenCV array, folder, glob, URL, webcam…
                    **data,
                    project=WorkDir,
                    save= PP.YOLOSave,                                         # save annotated images to `./runs`
                    name = foldername_save,
                    exist_ok= True,                                            # overwrite instead of incrementing
                    show=False,                                                # pop up a window with boxes
                    verbose=PP.YOLOPrint,                                      # console message control.
                    line_width=1)                                              # box border thickness in pixel.
                
                res = results[0]                                               # First result.
                outTxt = SaveDir + os.sep + stem + '.txt'
                if PP.SegmentationSaveJsonOption.lower() == 'labelme':
                    outJson = WorkDir + os.sep + stem + '.json'
                else:
                    outJson = SaveDir + os.sep + stem + '.json'
                mask_img = yolo2mask(res,outMask)                              # Converting results to mask.
                yolo2txt(res,outTxt)                                           # Converting results to txt.
                yolo2json(res,outJson)                                         # Converting results to json.
    
            # Writing the mask to local.
            if np.all(mask_img == 0): 
                print(formatSpec9 %(cname,str(i+1), str(nf), fname,(i+1)/nf*100))
            else:
                print(formatSpec10 %(cname,str(i+1), str(nf), fname,(i+1)/nf*100))
                     
    t_end = time.time()
    tt = t_end - t_start
    if nf == 0: raise ValueError(f'Warning: no photos found in {WorkDir}, please check...')
    print(formatSpec11 %(tt,tt/nf))
    print(formatSpec0)
    print()
    
    return 

#%% Get background filter data for folders.                                   #
def backgroundLevel2(PP):
    
    # Checking if working diretory has been specified.
    if PP.Directory == '': raise ValueError('Working directory is not defined.')
    
    # Pass parameters from PP to local.
    DirectoryL2 = PP.Directory 
    folders = getFolders(DirectoryL2,PP.ExcludingFolders)
    PP1 = copy.deepcopy(PP)
    for f in folders:
        PP1.Directory = DirectoryL2 + os.sep + f
        backgroundLevel1(PP1)
    return

#%% Generating background data based on given parameters.
def background(PP):
    formatSpec0 = 65*'='    
    print(formatSpec0)
    if PP.FolderLevel.lower() == 'l1':
        backgroundLevel1(PP)
    elif PP.FolderLevel.lower() == 'l2':
        backgroundLevel2(PP)
    else:
        raise 'Invalid metadata level option'
    return 

#%% Re-orientation image if necessary.
def reOrientation(image,exif):
    if 'Orientation' in exif:
        if exif['Orientation'] == 3:
            image = imutils.rotate(image, angle=180)
        elif exif['Orientation'] == 6:
            image = imutils.rotate(image, angle=270)
        elif exif['Orientation'] == 8:
            image = imutils.rotate(image, angle=90)
        #exif['Orientation'] = 1
        print(exif['Orientation'])
    return image, exif

#%% Modified on 3/36/2025: increasing processing speed.                       #
def cutPV(WorkDir,cut_size_w=1280,cut_size_h=1280,nframe=120, \
          writeVideoImage=True,overwrite=False,subdir = '',\
              imgfmt ='.png', video2imgonly=False, compression=1, compressionflag=False, maxframe=100000):
    
    timestart=time.time()                                                      # Start time of the code.
    formatSpec0 = '='*80
    #formatSpec1 = 'Cuts photos output folder: %s'
    formatSpec2 = 'Video photo output folder: %s'
    formatSpec3 = '%s: get frame %g from %g/%g in %.2f s (%3.2f%%)'
    formatSpec3b = '%s: get frame %g from %g/%g in %.2f s (%3.2f%%), file existed, skipping...'
    formatSpec4 = 'Computation done, elapsed time %.2f seconds.\n'

    print(formatSpec0)
    save_path = os.sep.join(WorkDir.split(os.sep)[0:-1])                       # The upper level path of the given working directory.
    cut_save_root = save_path + os.sep + subdir                                # Master output folder path.
    #v2p_save_root = save_path + os.sep + imgfmt[1:].upper()
    v2p_save_root = save_path
    #print(formatSpec1 %(cut_save_root))
    if not os.path.exists(cut_save_root) and subdir != '': os.mkdir(cut_save_root)# Create a folder to save all data.
    
    flag = True
    files,stems,ext,ispv = getFiles(WorkDir)
    for k in range(len(files)):
        fname = files[k]
        cut_save_k = cut_save_root + os.sep + stems[k]
        pv_name = WorkDir + os.sep + fname
        vpath = cut_save_k + '_V.csv'
        ppath = cut_save_k + '_P.csv'
        
        # Processing for videos.
        Pre = 'Cutting '+ fname
        if ispv[k] == 1:
            if flag: print(formatSpec2 %(v2p_save_root)); flag = False
            if not os.path.exists(v2p_save_root) : os.mkdir(v2p_save_root)     # Create a folder if not existed.
            if os.path.exists(vpath) and not overwrite: 
                print(Pre+': cuts existed, skip.'); continue
            else:
                print(Pre+': to size '+str(cut_size_w) + '(w) * ' + str(cut_size_h) + '(h) every '+ str(nframe)+' frames.')
            if os.path.exists(vpath) and not overwrite: print('Cuts existed, skipping.'); continue
            vidcap = cv2.VideoCapture(pv_name)                                                                     
            tframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))                 # Total frame number of video.
            nf = math.floor(tframe/nframe)                                     # Total output frame number.
            w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))                      # Width of video.
            h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))                     # Height of video.
            img_size = (h,w,3)                                                 # Video image size.
            cut_w = math.ceil(img_size[1]/cut_size_w)
            cut_h = math.ceil(img_size[0]/cut_size_h)
            bc = np.zeros((cut_w*cut_h*nf,6))
            
            for count in range(nframe, tframe+1, nframe):                      # Frame id from 1 to tframe (consistant with DJI index).
                tsi = time.time()                                              # Start time of the code.
                pv_out = v2p_save_root + os.sep + stems[k] +"_"+ str(count) + imgfmt
                ic = int(count/nframe)                                         # Current photo number.
                if ic>maxframe: continue
            
                if compressionflag: pv_out = v2p_save_root + os.sep + stems[k] +\
                    "_"+ str(count) + '_' + str(compression) + imgfmt
                
                flag = False
                if os.path.isfile(pv_out) and not overwrite:
                    if not video2imgonly:
                        image = cv2.imread(pv_out, cv2.IMREAD_UNCHANGED)
                    success = True
                    flag = True
                else:
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, count-1)               # Reading only the frame needed.
                    success, image = vidcap.read()                             # Only the selected frame is read and decoded
                    if success and writeVideoImage: 
                        if os.path.isfile(pv_out): os.remove(pv_out)
                        if compression == 1:
                            cv2.imwrite(pv_out,image)
                        else:
                            cv2.imwrite(pv_out,image,[cv2.IMWRITE_PNG_COMPRESSION, compression])
                            
                if success and not video2imgonly:
                    icut = 0
                    for j in range(cut_h):
                        for i in range(cut_w):
                            icut += 1
                            x0 = i*cut_size_w
                            if i == cut_w - 1: x0 = img_size[1] - cut_size_w
                            x1 = x0 + cut_size_w
                            y0 = j*cut_size_h
                            if j == cut_h - 1: y0 = img_size[0] - cut_size_h
                            y1 = y0 + cut_size_h
                            bc[icut-1 + (ic-1)*cut_w*cut_h,:] = [count,icut,x0,x1,y0,y1]
                            cut_save = cut_save_k +"_"+ str(count) + "_"+str(icut) + imgfmt
                            if (not os.path.exists(cut_save)) or overwrite:
                                img_cut = image[y0:y1,x0:x1]
                                if compression == 1:
                                    cv2.imwrite(pv_out,image)
                                else:
                                    cv2.imwrite(cut_save,img_cut,[cv2.IMWRITE_PNG_COMPRESSION, compression])
                                    
                    # Writing cut location to file.
                    fmt = ",".join(["%d"] * (bc.shape[1]))
                    np.savetxt(vpath,bc,fmt = fmt,delimiter=",")
                            
                tei = time.time()                                         
                if flag:
                    print(formatSpec3b %((fname,ic, count, tframe, tei - tsi, count/tframe*100)))
                else:
                    print(formatSpec3 %((fname,ic, count, tframe, tei - tsi, count/tframe*100)))

            vidcap.release()
            print('')
            
        elif ispv[k] == 0:
            pstem = '_'.join(stems[k].split('_')[0:-1])
            if pstem in stems: 
                print(Pre+': video photo, skip.'); continue
            elif os.path.exists(ppath) and not overwrite:
                print(Pre+': cuts existed, skip.'); continue
            else:
                print(Pre+': to size '+str(cut_size_w) + '(w) * ' + str(cut_size_h) + '(h).')
            
            # Loading image and re-orientating if necessary.
            image = cv2.imread(pv_name, cv2.IMREAD_UNCHANGED)
            exif = getExif(pv_name)
            if ('Orientation' in exif) and exif['Orientation'] in [3,6,8]:\
                image = reOrientation(image, exif)
                
            img_size = np.shape(image)
            cut_w = math.ceil(img_size[1]/cut_size_w)
            cut_h = math.ceil(img_size[0]/cut_size_h)
            bc = np.zeros((cut_w*cut_h,6))
            icut = 0
            for j in range(cut_h):
                for i in range(cut_w):
                    icut += 1
                    x0 = i*cut_size_w
                    if i == cut_w - 1: x0 = img_size[1] - cut_size_w
                    x1 = x0 + cut_size_w
                    y0 = j*cut_size_h
                    if j == cut_h - 1: y0 = img_size[0] - cut_size_h
                    y1 = y0 + cut_size_h
                    bc[icut-1,:] = [0,icut,x0,x1,y0,y1]
                    cut_save = cut_save_k + "_"+ str(icut)+imgfmt
                    img_cut = image[y0:y1,x0:x1]
                    if compression == 1:
                        cv2.imwrite(cut_save,img_cut)
                    else:
                        cv2.imwrite(cut_save,img_cut,[cv2.IMWRITE_PNG_COMPRESSION, compression])
                        
            fmt = ",".join(["%d"] * (bc.shape[1]))
            np.savetxt(ppath,bc,fmt = fmt,delimiter=",")
            
    timeend = time.time()                                                      # End time of the code.
    tt = timeend-timestart
    print(formatSpec4 %(tt))
    print(formatSpec0)
    
#%% Extracting photos from vidoes or copying photos from another folder local.%
def getPhotos(WorkDir,nframe=120,overwrite=False):
    folders = getFolders(WorkDir)
    for folder in folders:
        VPDir = os.path.join(WorkDir, folder)
        key = str(folder).lower()
        if key == 'videos':
            cutPV(VPDir, nframe = nframe, overwrite=overwrite, video2imgonly=True,\
                  compression=1, compressionflag=False, maxframe=10000000)            
                
        elif key == 'photos':
            files, _, exts, isv = getFiles(VPDir)
            basenames = [os.path.basename(f) for f in files]
            files_jpg    = [f for f in basenames if f.lower().endswith(('.jpg', '.jpeg'))]
            files_nonjpg = [f for f in basenames if not f.lower().endswith(('.jpg', '.jpeg'))]
            nonjpg_stems = {os.path.splitext(f)[0].lower() for f in files_nonjpg}
            for fname in files_nonjpg:
                src = os.path.join(VPDir, fname)
                dst = os.path.join(WorkDir, fname)
                if overwrite or not os.path.isfile(dst):
                    shutil.copy2(src, dst)
                    
            for fname in files_jpg:
                stem = os.path.splitext(fname)[0].lower()
                dst  = os.path.join(WorkDir, fname)
                if (overwrite or not os.path.isfile(dst)) and (stem not in nonjpg_stems):
                    photoCorrectionSingle(VPDir, WorkDir, fname)
    print('')
  
#%% Resizing photos to desired sizes. Correct photo orientation if needed.    #
def photoResize(inputdir,desiredsize=12_000_000,flag='resized', overwrite=False,
                exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")):
    
    outputdir = inputdir + '_' + flag
    os.makedirs(outputdir, exist_ok=True)
    
    all_images = [f for f in Path(inputdir).glob("*") if f.suffix.lower() in exts]
    nt = len(all_images)
    count = 1
    for img_path in all_images:
        # Get file name.
        filename = img_path.name
        output_path = Path(outputdir) / filename
        if output_path.is_file() and not overwrite:
            print(f"{count}/{nt}: {filename}, file existed, skipping...")
        else:
            img_pil = Image.open(img_path)
            
            # Get exif info of photos.
            exif_data = img_pil.info.get("exif", None)
            exif_dict = piexif.load(exif_data) if exif_data else None
            corrected = False
            if exif_dict:
                orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)
                if orientation != 1:
                    img_pil = ImageOps.exif_transpose(img_pil)
                    exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                    corrected = True
                    
            # Convert to cv.
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            h0, w0 = img_cv.shape[:2]
            orig_pixels = w0 * h0
        
            # Resize if needed.
            if orig_pixels < desiredsize:
                scale = math.sqrt(desiredsize / orig_pixels)
                wn = int(math.ceil((w0 * scale) / 32) * 32)
                hn = int(math.ceil((h0 * scale) / 32) * 32)
                img_cv = cv2.resize(img_cv, (wn, hn), interpolation=cv2.INTER_CUBIC)
                mpn = wn * hn /1e6
                if corrected:
                    print(f"{count}/{nt}: {filename}, orientation corrected, {w0}×{h0} → {wn}×{hn} ({mpn:.2f} MP)")
                else:
                    print(f"{count}/{nt}: {filename}, orientation unchanged, {w0}×{h0} → {wn}×{hn} ({mpn:.2f} MP)")
            else:
                wn, hn = w0, h0
                mpn = desiredsize / 1e6
                if corrected:
                    print(f"{count}/{nt}: {filename}, orientation corrected, size ≥ {mpn:.1f} MP, skipping...")
                else:
                    print(f"{count}/{nt}: {filename}, orientation unchanged, size ≥ {mpn:.1f} MP, skipping...")
        
        
            # Convert back to PIL for saving with EXIF.
            img_out = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
            # Save with EXIF.
            if exif_dict:
                exif_bytes = piexif.dump(exif_dict)
                img_out.save(output_path, exif=exif_bytes, quality=100)
            else:
                img_out.save(output_path, quality=100)
        count+= 1
        
#%% Performing qualicty control of grain size data.
def qualityControl(casedir,datatype='pv',videos=None,circularitypercenterror=5,
                   minimumgrainnumber=100, minimumdroneheight=1.5,startindex=0,coeff=None, 
                   writedata = True,threshold=0.35, zeroby = 'Z',
                   version='V3',modelname='YOLO11m.1280.20250322'):
    
    # Output information format control.
    fmt0  = '\nQuality control summary for %s:'
    fmt1a  = '%d/%d: %s, time %3.1f s, area %3.1f m2, valid area %3.1f m2 (%3.1f%%), frame %d, valid frame %g (%3.1f%%)'
    fmt1b  = '%d/%d: ***, grain number %d, number per image %3.1f; efficiency %3.1f m2/s, %3.1f grains/s'
    fmt2a  = 'Videos: time %3.1f s, area %3.1f m2, valid area %3.1f m2 (%3.1f%%), frame %d, valid frame %d (%3.1f%%)'
    fmt2b  = 'Videos: grain number %d, number per image %3.1f; efficinecy %3.1f m2/s, %3.1f grains/s'
    fmt3a  = 'Photos: time %3.1f s, area %3.1f m2, valid area %3.1f m2 (%3.1f%%), frame %d, valid frame %d (%3.1f%%)'
    fmt3b  = 'Photos: grain number %d, number per image %3.1f; efficiency %3.1f m2/s, %3.1f grains/s'
    fmt4a  = 'All: time %3.1f s, area %3.1f m2, valid area %3.1f m2 (%3.1f%%), frame %d, valid frame %d (%3.1f%%)'
    fmt4b  = 'All: grain number %d, number per image %3.1f; efficiency %3.1f m2/s, %3.1f grains/s'
    fmt5  = 'No data passing quality control, skipping data writing.'
    fmt6 = 65*'='
    
    # Local keywords.
    vkey = 'Statistics'
    #skey = 'Scales'
    coeff0 = [-2.59e-5, 2.16e-3, 0.278, 0.0219]
    coeffp = coeff if coeff is not None else coeff0
    res0 = np.polyval(coeffp, minimumdroneheight)
    
    # Processing directory info.
    if os.path.isfile(casedir):
        stpath = casedir
        stname =  os.path.basename(casedir)
        casedir = os.path.dirname(casedir)
        fname = os.path.basename(casedir)
    else:
        fname = os.path.basename(casedir)
        stname = '_'.join([vkey,version,fname,modelname,str(int(threshold*100))]) + '.csv'
        stpath = os.path.join(casedir,stname)
    gsd = pd.read_csv(stpath,keep_default_na=False)                            # Loading pre-generated grain size data.
    gsd['Site_ID'] = gsd['Site_ID'].astype(str).str.strip()
    print(fmt0 %(fname))
    
    # Processing for video data.
    vfolder = os.path.join(casedir,'videos')
    nva = 22
    stv = -9999*np.ones((0, nva))
    stva = -9999*np.ones((1, nva))
    gsdv = pd.DataFrame()
    n00vs = 0
    typeid = 0
    if 'v' in datatype and os.path.isdir(vfolder):
        if videos is None: 
            videos, _, _, ispv = getFiles(vfolder)                             # Obtain video names if not provided.
            videos = [v for v, p in zip(videos, ispv) if p == 1]
        nv = len(videos)                                                       # Number of videos.
        drs, g0vs, a0vs, a2vs, n0vs, n1vs, n2vs = [0] * 7
        for i in range(nv):
            vname = videos[i]
            vpath = os.path.join(vfolder,str(videos[i]))
            info = readEXIF(vpath)
            wv = info['Width']
            hv = info['Height']
            dmv = wv*hv/1e6
            fr = info['FrameRate']
            drv = info['Duration']
            vid = vname.split('.')[0].split('_')[1]
            if not str(vid).strip().isdigit(): 
                raise ValueError("Video index is not numeric...")
            vidi = int(vid)
            nv0 = len(vid) - len(vid.lstrip('0'))
            if nv0>0: vid = zeroby * nv0 + vid[nv0:]
            gsdi = gsd[gsd.Site_ID == vid]
            n00v = gsdi.shape[0]
            gsdi = gsdi[gsdi['L1L2_res_meter_per_pixel']>0]
            n0v = gsdi.shape[0]
            a0v = np.sum(gsdi['Area_m2'])
            g0v = np.sum(gsdi['Grain_number_threshold'])
            if 'DJI' in vname or '':
                gsdi = gsdi[gsdi['L1L2_res_millimeter_per_pixel']>res0]
            else:
                gsdi = gsdi[abs(gsdi['Relative_error_percent'])<=circularitypercenterror]
            n1v = gsdi.shape[0]
            gsdi = gsdi[gsdi['Grain_number_threshold']>minimumgrainnumber]
            gsdv = pd.concat([gsdv, gsdi], ignore_index=True)
            n2v = gsdi.shape[0]
            a2v = np.sum(gsdi['Area_m2'])
            rav = a2v/a0v
            rav, r0v, r1v, r2v =  a2v/a0v, n0v/n00v, n1v/n00v, n2v/n00v
            eav, env = a0v/drv, g0v/drv                                        # Efficiency.
            gnv = g0v/n0v                                                      # Grain number per frame.
            sti = np.array([typeid,i+startindex,vidi,fr,wv,hv,dmv, drv,g0v,a0v,a2v,rav, 
                            n00v,n0v,n1v,n2v, r0v, r1v, r2v, eav,env,gnv])
            drs+= drv
            g0vs+= g0v
            a0vs+= a0v
            a2vs+= a2v
            n00vs+= n00v
            n0vs+= n0v
            n1vs+= n1v
            n2vs+= n2v
            stv = np.vstack((stv, sti))
            stva[0,4:7] =  stv[-1,4:7]
            stva[0,8:] =  stv[-1,8:]
            print(fmt1a %(i+1,nv,vname,drv,a0v,a2v,rav*100,n00v,n2v,r2v*100))
            print(fmt1b %(i+1,nv,g0v,gnv,eav,env))
        if nv>1:
            stvf = -9999*np.ones((1, nva))
            stvf[0,0] = typeid
            stvf[0,3:7] = np.mean(stv[:,3:7],axis=0)
            ravs, r0vs, r1vs, r2vs =  a2vs/a0vs, n0vs/n00vs, n1vs/n00vs, n2vs/n00vs
            eavs, envs = a0vs/drs, g0vs/drs
            gnvs = g0vs/n0vs
            stvf[0,7:] = np.array([drs,g0vs,a0vs,a2vs,ravs, 
                            n00vs,n0vs,n1vs,n2vs, r0vs, r1vs, r2vs, eavs,envs, gnvs])
            stva = stvf
            stv = np.vstack((stv,stvf))
            print(fmt2a %(drs,a0vs,a2vs,ravs*100,n00vs,n2vs,r2vs*100))
            print(fmt2b %(g0vs,gnvs,eavs,envs))
            
    # Processing for photo data.
    stp =  -9999*np.ones((1, nva))
    gsdp = pd.DataFrame()
    n00p = 0
    eps = 0
    drp = eps
    stp[0,0] = 1
    if 'p' in datatype:
        gsdi = gsd[gsd.Photo_indicator == 1]
        if gsdi.shape[0]>0:
            n00p = gsdi.shape[0]
            gsdi = gsdi[gsdi['L1L2_res_meter_per_pixel']>0]                    # QC0.
            n0p = gsdi.shape[0]
            a0p = np.sum(gsdi['Area_m2'])
            g0p = np.sum(gsdi['Grain_number_threshold'])
            t = pd.to_datetime(gsdi["Date_PST"])
            drp = (t.max() - t.min()).total_seconds()     
            
            w = np.mean(gsdi['Width_pixel'])
            h = np.mean(gsdi['Height_pixel'])
            dm = np.mean(gsdi['Width_pixel']*gsdi['Height_pixel'])/1e6 
            mask = gsdi["Name"].str.contains("DJI", case=False, na=False)
            gsdi1  = gsdi[mask]                                                # Drone photos.
            gsdi2  = gsdi[~mask]                                               # None-drone photos.
            gsdi1 = gsdi1[gsdi1['L1L2_res_millimeter_per_pixel']>res0]         # QC1: drone.
            gsdi2 = gsdi2[abs(gsdi2['Relative_error_percent'])
                          <=circularitypercenterror]                           # QC1: smartphone.
            gsdi = pd.concat([gsdi1, gsdi2], ignore_index=True)
            n1p = gsdi.shape[0]
            gsdi = gsdi[gsdi['Grain_number_threshold']>minimumgrainnumber]     # QC2.
            gsdp = gsdi
            n2p = gsdi.shape[0]
            a2p = np.sum(gsdi['Area_m2'])
            rap = a2p/a0p
            rap, r0p, r1p, r2p =  a2p/a0p, n0p/n00p, n1p/n00p, n2p/n00p
            eap, enp = [0] * 2
            if drp >0: eap, enp = a0p/drp, g0p/drp
            gnp = g0p / n0p
            stp[0,[4,5,6]] = np.array([w,h,dm])
            stp[0,7:] = np.array([drp,g0p,a0p,a2p,rap, n00p,n0p,n1p,n2p, r0p, r1p, r2p, eap,enp, gnp])
            print(fmt3a %(drp,a0p,a2p,rap*100,n00p,n2p,r2p*100))
            print(fmt3b %(g0p,gnp,eap,enp))
            
    # Final data.
    gsdqc = pd.concat([gsdv, gsdp], ignore_index=True)
    stqc = stv if drp == eps else np.vstack((stv,stp))
    if n00vs>0 and n00p>0:
        stvp = -9999*np.ones((1, nva))
        stvp[0,0] = 2
        stvp[0,4:7] =  (stva[0,4:7]*n00vs + stp[0,4:7]*n00p)/(n00vs+n00p)
        g0 = g0p + g0vs
        a0 = a0p + a0vs
        a2 = a2p + a2vs
        n00 = n00p + n00vs
        dr = drp + drs
        n0 = n0p + n0vs
        n1 = n1p + n1vs
        n2 = n2p + n2vs
        ra, r0, r1, r2 =  a2/a0, n0/n00, n1/n00, n2/n00
        ea, en = [0] * 2
        if dr >0: ea, en = a0/dr, g0/dr
        gn = g0/n0
        stvp[0,7:] = np.array([dr,g0, a0,a2,ra,n00,n0,n1,n2, r0, r1, r2, ea, en,gn])
        print(fmt4a %(dr,a0,a2,ra*100,n00,n2,r2*100))
        print(fmt4b %(g0,gn,ea,en))
        stqc = np.vstack((stqc,stvp))
    
    # Converting data to dataframe.
    cnames = ['Data_type','Video_ID','Video_name','Frame_rate_fps','Width_px','Height_px',
              'Dimension_megapixel','Duration_s','Grain_number_qc0','Area_qc0_m2',
              'Area_qc2_m2','Area_ratio_qc2','Total_frame','Frame_qc0','Frame_qc1',
              'Frame_qc2','Frame_ratio_qc0','Frame_ratio_qc1','Frame_ratio_qc2',
              'Efficiency_m2_per_second','Efficiency_grains_per_second','Grain_number_per_frame']
    stqc = pd.DataFrame(stqc, columns=cnames)
    stqc.insert(0, 'Folder', fname)
    
    # Output data.
    if gsdqc.shape[0] == 0: print(fmt5)
    
    qkey1 = 'QC'
    qkey2 = 'QCSummary'
    if writedata and gsdqc.shape[0] > 0:
        gsdqcpath = casedir + os.sep + qkey1 + stname
        stqcpath =casedir+os.sep+qkey2+ '_' + '_'.join(stname.split('_')[1:])
        gsdqc.to_csv(gsdqcpath, index=False)
        stqc.to_csv(stqcpath, index=False)
    print(fmt6)
    
    return gsdqc, stqc

#%%