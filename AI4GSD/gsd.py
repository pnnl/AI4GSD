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
import os, time, cv2, requests, copy, re, platform, fnmatch, matplotlib, glob
import numpy as np
import pandas as pd                                                         
from PIL import Image
from pathlib import Path
from natsort import natsorted
from shapely.geometry import box                                               # Computing union area.
from shapely.ops import unary_union
from .ultilities import agent, defaultResults                                  # User-defined package.
from .mathm import getFiles, ecdfm, center2corners
from .mathm import __file__ as mathm_path
#from memory_profiler import profile                                           # Only for memeory analysis.

#%% Extracting the individual sediment sizes, distributions, & characteristic #
# grain sizes as well as their plottings from pre-trained AI models hosted    #
# in cloud serveres. The AI prediction is conducted on the cloud, but the     #
# post-processing of data are performed on local machines.                    #
#@profile
def Photo2GSDSingle(PP):        
    
    # Printing information control.
    #formatSpec0 = 65*'='
    #formatSpec1 = 'AI4GSD: an AI and cloud powered tool for grain size quantification'
    formatSpec2 = 'Model name: %s from %s for %s detection'
    formatSpec2a = 'Model layer: %s, model parmeter: %4.2fM, submodule number: %s'
    formatSpec3 = 'IOU: %3.2f, YOLO confidence: %3.2f, user confidence: %3.2f'
    formatSpec5 = 'Folder name: %s'
    formatSpec6 = 'Output folder: %s'
    formatSpec7 = 'Number of photos: %s, number of exluded photos %s'
    formatSpec8 = 'Found existing summary statistics file: %s, skipping...'
    formatSpec9 = '%s/%s: %s, resolution %3.2f %s/px (%3.2f%%), %3.2f s'
    formatSpec10 = 'Found existing grain size file: %s, skipping...'
    formatSpec11 = 'Re-use data from model %s' 
    formatSpec12 = 'Detected %s %ss, diagonal and count based D10, D50, D60, and D84 are: %4.2f, %4.2f, %4.2f, and %4.2f %s' 
    formatSpec13 = 'Detected %s %ss, diagonal and area  based D10, D50, D60, and D84 are: %4.2f, %4.2f, %4.2f, and %4.2f %s'
    formatSpec14 = 'Output folder summary data to: %s'
    formatSpec15 = 'Total number of grains: %d, number of grains per photo: %3.2f'
    formatSpec16 ='Total time: %3.2f s, time per photo %3.2f s'
    #formatSpec17 = 65*'-' +'\n' 

    #print(formatSpec0)
    #print(formatSpec1)
    t_start = time.time()
    
    # Pass latest parameters into local code.
    PP.ObjectName = 'rock'
    WorkDir = PP.Directory
    model_name = PP.ModelName
    obj_name = PP.ObjectName
    SCName = PP.ScaleTemplateName
    GSDName = PP.StatisticsTemplateName
    scales_d = PP.ScalesDefault
    scales_u = PP.Scales
    pext = PP.TargetPhotoExtension
    exludes = PP.ExcludingPhotos
    outkeyword = PP.OutputKeyWord
    validationkeyword = PP.ValidationPlotKeyWord
    pr = PP.FloatPrecision
    refdate = PP.DefaultDate
    zmax = PP.NoValueIndicator
    templatefolder = PP.TemplateFolderName
    FolderNameBeforeThreshold = PP.SaveYOLODataBeforeThresholdFolder
    OverWriteGSDFilePhoto = PP.OverWriteGSDFilePhoto
    OverWriteGSDFileFolder = PP.OverWriteGSDFileFolder
    OverWrtieAll = PP.OverWriteAll
    printimgsize = PP.PrintImageSize
    scalefactor = PP.ImageSizeFactor                                           # Used to control image size for inference.
    segFolder = PP.SegmentationSaveFolderName
    llw = round(PP.OverlayLableLineWidth)                                      # Has to be integer.
    conf0 = PP.ConfidenceThresholdYOLO                                         # Default YOLO confidence threshold.
    conf1 = PP.ConfidenceThresholdUser                                         # User confidence threshold.
    iou = PP.YOLOIOU                                                           # Default YOLO iou.
    usemask = PP.UseMask                                                       # Control if using Anthro filter mask.
    gsdflag = PP.GSDFlag
    overlap_photo = PP.GrainBackgrouondOverlapThreshold              
    overlap_grain = PP.InvividualGrainBackgrouondOverlapThreshold
    segalpha = float(PP.SegmentationOverlayTransparency)                       # Transparency of segementation.
    line_color1_rgb = PP.SaveOverlayGrainColor                     
    line_color2_rgb = PP.SaveOverlayScaleColor                    
    line_color3_rgb = PP.SaveOverlayBackgroundColor                      
    grainfolder = PP.GrainAIPredictionSaveFolderName
    outYOLOFolderBeforeOld = PP.UserYOLOFolder
    notfromdata = outYOLOFolderBeforeOld == ''
    szmax = PP.MaximumGrainSize                                                # Remove grains larger than this.
    szmin = PP.MinimumGrainSize                                                # Remove grains smaller than this.
    
    # Update model name if using existing data, which may not come from a model.
    model_name += ".0.0" if len(model_name.split('.')) == 1 else ""
    
    # Adjust confidence threshold.
    conf0 = min(conf0, conf1)                                                  # Conf0 is always less than conf1.
    
    # Define current folder name and output folder path.
    MainDir = os.path.dirname(WorkDir)
    MainDir = WorkDir
    foldername = os.path.basename(WorkDir)
    foldername_save = foldername + '_' + model_name + '_' + "{:.0f}".format(conf1*100)
    if gsdflag != '': foldername_save = foldername_save + '_' + gsdflag
    PP.SaveFolder = foldername_save
    SaveDir = MainDir + os.sep + grainfolder + os.sep + foldername_save
    os.makedirs(SaveDir,exist_ok=True)
    
    # Define folder name for writing YOLO data before thresholding.
    outYOLOFolderBefore = SaveDir + os.sep + FolderNameBeforeThreshold

    # Defining cloud information.
    cloud_models=['RepPoints37M','YOLO11m.1280.20250127','YOLO11x.1280','YOLO11m.640',
                  'YOLOv5x6u.1280','YOLO11m.1280.20250307','YOLO11m.640.20250307',
                  'YOLO11m.1280.20250308','YOLO11m.640.20250309','YOLO11m.1280.20250309',
                  'YOLO11m.1280.20250319','YOLO11m.1280.20250320',
                  'YOLO11x.1280.20250320','YOLO11m.1280.20250321',
                  'YOLO11m.1280.20250322','YOLO11m.1280.20250926',
                  'YOLO11m.2560.20250927']
    # if any(model_name.lower() == item.lower() for item in cloud_models):
    #     cloud_name, api_key, model_id = cloud(PP)
    cmpt = PP.ComputingSource.lower()
    if any(model_name.lower() == item.lower() for item in cloud_models):
        cloud_name, api_key, model_id = agent(PP)
        
        # Loading YOLO model
        if cmpt == 'local': 
            from ultralytics import YOLO
            model = YOLO(str(Path(model_id)))
            
            import torch.nn as nn
            modules = list(model.model.modules())
            nmodules = len(modules)
            npp = sum(p.numel() for p in model.model.parameters())/1e6
            nl = sum(isinstance(m, nn.Conv2d) for m in modules)
            
    print(formatSpec5 %(foldername))
    print(formatSpec6 %(foldername_save))
    print(formatSpec2 %(model_name, cmpt, obj_name))
    if notfromdata and cmpt == 'local': print(formatSpec2a %(nl, npp,nmodules))
    print(formatSpec3 %(iou, conf0, conf1))

    # Loading scale template.
    codedir = os.path.dirname(mathm_path) + os.sep + templatefolder
    SCFile = os.path.join(codedir, SCName)
    if os.path.isfile(SCFile) and os.path.getsize(SCFile) >10:
        headers_sc = pd.read_csv(SCFile,keep_default_na=False)
        cnames = list(headers_sc.columns)
        idx1a = cnames.index('L1_px1_pixel')                               
        idx1b = cnames.index('Resolution_velocity_integrate_meter_per_pixel')
        idx2a = cnames.index('Make')
        idx2b = cnames.index('Notes')
        defaults = [['N/A']*idx1a + [zmax]*(idx1b - idx1a + 1) +\
            [refdate] + [zmax] + ['N/A']*(idx2b - idx2a + 1)]
        headers_sc = pd.DataFrame(defaults,columns=cnames)
    else:
        raise ValueError(f"Missing required scale template file {SCFile}.")
    
    # Loading statistics template.
    gsddir = os.path.dirname(mathm_path) + os.sep + templatefolder
    GSDFile = os.path.join(gsddir, GSDName)
    [GSDNameStem,gsdext] = GSDName.split('.')
    if os.path.isfile(GSDFile) and os.path.getsize(GSDFile) >10:
        headers_gsd = pd.read_csv(GSDFile,keep_default_na=False)
        cnames_gsd = list(headers_gsd.columns)
    else:
        raise ValueError(f"Missing required scale template file {GSDFile}.")
    
    # Loading pre-generated scale file.
    [SCNameStem,ext] = SCName.split('.')
    localSCName = SCNameStem + '_' + foldername + '.' + ext
    localSCFile = os.path.join(WorkDir,localSCName)
    scales_user = pd.DataFrame(columns=cnames)
    if os.path.isfile(localSCFile) and os.path.getsize(localSCFile) >10:
        scales_user = pd.read_csv(localSCFile,keep_default_na=False)
        scales_user = scales_user.dropna(how='all')
        
    # Converting the use input scales into a dataFrame with 49 columns.
    scales_u = scales_d if scales_u is None else {**scales_d, **scales_u}
    cnames2 = ["Name", "L1L2_res_meter_per_pixel"]
    cnames47 = list(set(cnames) - set(cnames2))
    scales_input = pd.DataFrame(list(scales_u.items()), columns=cnames2)
    idt = cnames.index('Date_PST')
    scales_input[cnames[idt:idt+1]] = refdate
    scales_input[cnames[idt+2:]] = 'N/A'  
    scales_input['Folder'] = foldername_save
    scales_input['Scale_source'] = 'N/A'
    cnames39 = list(set(cnames47) - set(cnames[idt:idt+1]) - \
                    set(cnames[idt+2:]) - set(['Folder','Scale_source']))
    scales_input[cnames39] = zmax
    scales_input = scales_input[cnames]
    mask = scales_input["L1L2_res_meter_per_pixel"] > 0
    scales_input.loc[mask, "L1L2_res_millimeter_per_pixel"] = \
    scales_input.loc[mask, 'L1L2_res_meter_per_pixel'] * 1000
    scales_input.loc[~mask,'L1L2_res_meter_per_pixel'] = -1                    # Set default resolution as -1 m/pixel.
    scales_input.loc[~mask,'L1L2_res_millimeter_per_pixel'] = -1
 
    # Merging user input scales and metadata scales into a combined one.
    scales_all = scales_input if len(scales_user) == 0 else \
        pd.concat([scales_input, scales_user], axis=0, ignore_index=True)
    
    # Removing rows with non-positive resolution and removing repeated rows.
    scales_all_copy = scales_all.copy()
    scales_all_copy = scales_all_copy.drop_duplicates(subset='Name', keep='first')
    scales_all = scales_all[scales_all['L1L2_res_meter_per_pixel']>=0]
    scales_all = scales_all.drop_duplicates(subset='Name', keep='first')
    scales_names = scales_all[cnames2[0]].values
    scales_index = {Path(filename).stem: i for i, filename in enumerate(list(scales_names))}
    
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
    
    # Checking if GSD file exists in SaveDir. 
    outGSDName = SaveDir + os.sep + GSDNameStem + '_' + foldername_save + '.csv'
    gsd = pd.read_csv(outGSDName,keep_default_na=False) if \
        os.path.isfile(outGSDName) and os.path.getsize(outGSDName) >10 else pd.DataFrame()
    nfg = len(gsd)
    
    # Skipping predictions if file exists and no new files.
    if nf == nfg and (not OverWriteGSDFileFolder) and not OverWrtieAll:
        print(formatSpec8 %(os.path.basename(outGSDName)))
        nf = 1
    else:
        # Reset GSD variable.
        gsd = pd.DataFrame()
        
        # Checking if results from the same model name has been generated.    #
        # We can directly load the YOLO data before threshold for cases with  #
        # different threshold. This saves time.                               #
        #msg = "Re-use model name not specified in model parameter PP!"
        if outYOLOFolderBeforeOld != '':
            model_name_source = PP.ModelName
        else:
            folders = [f.name for f in Path(MainDir).iterdir() if \
                        (f.is_dir() and foldername in f.name)]
            matches = [item for item in folders if re.search(model_name, item)]
            for folder in matches:
                outGSDNameExist = MainDir + os.sep + folder + os.sep + \
                    GSDNameStem + '_' + folder + '.csv'
                if os.path.isfile(outGSDNameExist):
                    outYOLOFolderBeforeOld = MainDir + os.sep + folder + \
                        os.sep + FolderNameBeforeThreshold
                    model_name_source = folder.split('_')[1]                   # Name of the model data to be reused.
                    break
                
        # Looping for each photo.
        results0 = defaultResults(PP)
        count = 0
        dt = 0
        rows = []

        # First write of the data.
        first_write = True
        if PP.SaveStatisticsData: open(outGSDName, "w").close()
        #files = files[1:2]
        for i in range(len(files)):
            ts = time.time()
            # Get photo name, parent name, and extension.                     #
            fname = files[i]
            ext = fname.split('.')[-1]
            stem = '.'.join(fname.split('.')[0:-1])
            
            # Define output file names.
            outLocalGSDName =  SaveDir + os.sep + stem + '_Statistics' + '.csv'
            outLocalSizeName = SaveDir + os.sep + stem + '.csv'
            outYOLOBeforeTxt = outYOLOFolderBefore + os.sep + stem + '.txt'
            outYOLOBeforeScoreTxt = outYOLOFolderBefore + os.sep + stem + '_p' + '.txt'
            outYOLOBeforeClassTxt = outYOLOFolderBefore + os.sep + 'classes' + '.txt'
            outYOLOAfterTxt =      SaveDir + os.sep + stem + '.txt'
            outYOLOAfterClassTxt = SaveDir + os.sep + 'classes' + '.txt'
            
            # If grain size data for photo i has been generated, then loading #
            # the data and moving to end of the code for the next photo.      #
            if os.path.isfile(outLocalGSDName) and (not OverWriteGSDFilePhoto) and not OverWrtieAll:
                gsd_i = pd.read_csv(outLocalGSDName,keep_default_na=False)
                resolution = gsd_i[cnames2[1]].values[0]
                nrs = gsd_i['Grain_number_threshold'].values
                dt = time.time() - ts
                unit = 'px' if resolution <0 else 'mm'
                sx = -1 if resolution <0 else 1000
                print(formatSpec9 %(str(i+1), str(nf), fname,sx*resolution,unit,(i+1)/nf*100,dt))
                print(formatSpec10 %(os.path.basename(outLocalGSDName)))
            else:
                #-------------------------------------------------------------#
                # Extract image size information.
                imgpath = WorkDir + os.sep + fname
                with Image.open(imgpath) as _im:
                    sw, sh = _im.size  
                #imgi = Image.open(imgpath).convert('RGB')
                #sw = imgi.size[0]                                             # True width of image (metadata).
                #sh = imgi.size[1]                                             # True height of image (metadata).
                
                # Define image size used for AI inference.
                img_type = ['default','image']
                img_type_err = ', '.join(img_type)
                if PP.ImageSizeType.lower() == 'default' or cmpt in 'cloud':   # img_size has to be single value for cloud inference.
                    img_size0 = int(int(model_name.split('.')[1])*scalefactor) # Single value 1280, becomes (1280x1280).
                    img_size = (img_size0,img_size0)
                elif PP.ImageSizeType.lower() == 'image':
                    scale_max = (PP.MaximumPhotoResolution / (sw * sh)) ** 0.5
                    scalefactor = min(scalefactor,scale_max)
                    img_size = (int(sw*scalefactor),int(sh*scalefactor))       # Width, height.
                else:
                    msg = f"Invalid image type: {PP.ImageSizeType}." + \
                        f"Options: {img_type_err}"
                    raise ValueError(msg)
                if printimgsize: print(f'Inference image size: {img_size}')
                
                # Determining photo resolution.
                if stem in scales_index.keys():
                    idx = scales_index[stem]                                   # Index of current photo in the scales file.
                    resolution = scales_all[cnames2[1]].values[idx]
                    notes = scales_all['Notes'].values[idx]
                else:
                    if 'all' in scales_index.keys(): 
                        idx = scales_index['all']
                        resolution = scales_all[cnames2[1]].values[idx]        # Using all resolution if resolution not defined.
                        notes = 'all'
                    elif 'none' in scales_index.keys():
                        idx = scales_index['none']
                        resolution = scales_all[cnames2[1]].values[idx]        # Using all resolution if resolution not defined.
                        notes = 'none'
                    else:
                        resolution = scales_u['none']                          # Using default resolution if resolution not provided.
                        notes = 'default'
                        
                # Access model, conduct predictions and post processing.
                unit0 = 'px' if resolution <0 else 'mm'
                sx = -1 if resolution <0 else 1000
                print(formatSpec9 %(str(i+1), str(nf), fname,sx*resolution,unit0,(i+1)/nf*100, dt))
                YOLOOld = outYOLOFolderBeforeOld + os.sep + stem + '.txt'
                YOLOOldScore = outYOLOFolderBeforeOld + os.sep + stem + '_p' + '.txt'
                
                # Loading existing data if existed, otherwise call the AI model
                # os.path.getsize(YOLOOld)>190 means it has at least 5 objects.
                if os.path.isfile(YOLOOld) and os.path.getsize(YOLOOld)>190 \
                    and PP.UserYOLOFolder != '':
                    print(formatSpec11 %(model_name_source))
                    labels_yolo_old = np.loadtxt(YOLOOld)                      # Loading existing YOLO data.
                    nrn = labels_yolo_old.shape[0]
                    nra = nrn
                    obj_score_output_old = \
                        np.stack((labels_yolo_old[:,0],np.ones(nrn)),axis=1)   # Setup defualt confidence score as 1.
                    
                    # Loading confidence socre if exists.
                    if os.path.isfile(YOLOOldScore):
                        obj_score_output_old = np.loadtxt(YOLOOldScore)        # Upadting score if existed.
                    
                    obj_score = obj_score_output_old[:,1]
                    obj_id = obj_score_output_old[:,0]
                    obj_x = labels_yolo_old[:,1]*sw
                    obj_y = labels_yolo_old[:,2]*sh
                    obj_w = labels_yolo_old[:,3]*sw
                    obj_h = labels_yolo_old[:,4]*sh
                    obj_px = center2corners(labels_yolo_old[:,1:3],labels_yolo_old[:,3:5])
                    obj_px = obj_px*[sw,sh,sw,sh]
                else:
                    max_det = PP.MaximumNumberOfObject
                    data = {"imgsz": img_size, "conf": conf0, "iou": iou,"max_det":max_det}
                    if cmpt == 'cloud':
                        if cloud_name.lower() == 'ultralytics':
                            headers = {"x-api-key": api_key}
                            with open(imgpath, "rb") as f:
                                response = requests.post(model_id, headers=headers, data=data, files={"file": f})
                            
                            # Moving to the next predictin if status_code is not 200.
                            if response.status_code != 200: continue
                            results_ai = response.json()['images'][0]['results']
                            
                            # Creating a fake data when number of objects less than 2.
                            results=results0 if len(results_ai)<=2 else results_ai
                            
                            #obj_name = results[0]['name']
                            nrc = len(results)
                            obj_id = []; obj_px = []; obj_score = []
                            for k in range(nrc):
                                obj_id.append(results[k]['class'])
                                bb = [round(results[k]['box']['x1']),round(results[k]['box']['y1']),\
                                      round(results[k]['box']['x2']),round(results[k]['box']['y2'])]
                                obj_px.append(bb)
                                obj_score.append(results[k]['confidence'])
                                    
                        elif cloud_name.lower() == 'landingai':
                            # Only using LandingAI tool in Windows.
                            if PP.LandingAIInferenceMethod.lower() == 'landingai' and \
                                platform.system() == 'Windows':
                                from landingai.predict import Predictor
                                predictor = Predictor(model_id, api_key=api_key)
                                #results_ai = predictor.predict(imgi)
                                results_ai = predictor.predict(imgpath)
                                results=results0 if len(results_ai)<=2 else results_ai
                                
                                #obj_name = results[0].label_name
                                nrc = len(results)
                                obj_id = []; obj_px = []; obj_score = []
                                for k in range(nrc):
                                    obj_id.append(results[k].label_index)
                                    obj_px.append(results[k].bboxes)
                                    obj_score.append(results[k].score)
                            else:
                                headers = {'apikey': api_key}
                                url = 'https://predict.app.landing.ai/inference/' +\
                                'v1/predict?endpoint_id=' + model_id
                                with open(imgpath, "rb") as f:
                                    response = requests.post(url, headers=headers, files={"file": f})
                                
                                # Moving to the next predictin if status_code is not 200.
                                if response.status_code != 200: continue
                            
                                results_ai = response.json()['backbonepredictions']
                                results=results0 if len(results_ai)<=2 else results_ai
                                
                                keys = list(results.keys())
                                #obj_name = results[keys[0]]['labelName']
                                nrc = len(keys)
                                obj_id = []; obj_px = []; obj_score = []
                                for k in range(nrc):
                                    obj_id.append(results[keys[k]]['labelIndex'])
                                    bb = [results[keys[k]]['coordinates']['xmin'],
                                          results[keys[k]]['coordinates']['ymin'],
                                          results[keys[k]]['coordinates']['xmax'],
                                          results[keys[k]]['coordinates']['ymax']]
                                    obj_px.append(bb)
                                    obj_score.append(results[keys[k]]['score'])    
                        else:
                            raise 'Invalid cloud name'
                            
                        # Re-organizing data.
                        obj_px = np.asarray(obj_px)
                        obj_id = np.asarray(obj_id)
                        obj_id = obj_id - obj_id[0] + 0                        # Rename id.
                        obj_score = np.asarray(obj_score)
                        obj_x = (obj_px[:,0] + obj_px[:,2])/2.0
                        obj_y = (obj_px[:,1] + obj_px[:,3])/2.0
                        obj_w = obj_px[:,2] - obj_px[:,0]
                        obj_h = obj_px[:,3] - obj_px[:,1]
                        
                        # Wait for a few seconds to avoid over-loading cloud server. 
                        time.sleep(PP.SleepTime) 
                        nra = len(results_ai)
                        
                    elif cmpt == 'local':   
                        results = model.predict(
                            source=str(imgpath),                               # can be path, OpenCV array, folder, glob, URL, webcam…
                            **data,
                            project=foldername_save,
                            save= PP.YOLOSave,                                 # save annotated images to `./runs`
                            name = PP.GrainAIPredictionSaveFolderName,         # Where to save grain AI predictions.
                            exist_ok= PP.YOLOOverWrite,                        # overwrite instead of incrementing
                            show=False,                                        # pop up a window with boxes
                            verbose=PP.YOLOPrint,                              # console message control.
                            line_width=1)                                      # box border thickness in pixel.
                        
                        # Data format conversion.
                        #obj_names = []; 
                        obj_values  = []
                        for r in results:
                            for xyxy, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                                x1, y1, x2, y2 = map(int, xyxy)
                                #obj_names.append(model.names[int(cls)])
                                obj_values.append([
                                    x1, y1, x2, y2,
                                    int(cls),
                                    float(conf),
                                ])
                        del results
                                
                        nra = 0
                        if len(obj_values)>0:
                            #obj_names = np.array(obj_names, dtype=str)
                            obj_values  = np.array(obj_values, dtype=pr)
                            nra = obj_values.shape[0]
                            obj_px = obj_values[:,0:4]
                            obj_id = obj_values[:,4]
                            obj_id = obj_id - obj_id[0] + 0                    # Rename id.
                            obj_score = obj_values[:,5]
                            obj_x = (obj_px[:,0] + obj_px[:,2])/2.0
                            obj_y = (obj_px[:,1] + obj_px[:,3])/2.0
                            obj_w = obj_px[:,2] - obj_px[:,0]
                            obj_h = obj_px[:,3] - obj_px[:,1]
                            del obj_values
                            
                # Further calculation of other information based on xywh and p.
                pp00, pp0, pps = [zmax]*3
                nrn, nrs = [0]*2
                labels_r = None
                if nra>0:
                    pp00 = np.mean(obj_score)*100                              # Average accuracy before segementation.
                    nrn = len(obj_score)
                    obj_r = np.sqrt(obj_w**2 + obj_h**2)
                    labels_px = np.stack((obj_id,obj_x,obj_y,obj_w,obj_h),axis=1)
                    labels_yolo = labels_px.copy()
                    labels_yolo[:,[1,3]] = labels_yolo[:,[1,3]]/sw
                    labels_yolo[:,[2,4]] = labels_yolo[:,[2,4]]/sh
                    labels_r = labels_px.copy().astype(pr, copy=False)
                    area = labels_r[:,3]*labels_r[:,4]
                    area_r = area*((float(resolution))**2)
                    obj_others = np.stack((obj_r,area,np.ones((nrn))*abs(resolution),obj_score),axis=1)
                    labels_r = np.hstack((labels_r,obj_others))
                    labels_r[:,1:6] = labels_r[:,1:6]*abs(resolution)
                    labels_r[:,6] = labels_r[:,6]*resolution**2
                    obj_score_output = np.stack((labels_yolo[:,0],obj_score),axis=1)
                    
                    # Loading mask of non-sediments if available.
                    outMask = WorkDir + os.sep + segFolder + os.sep + stem + '.tif'
                    mask01 = [] 
                    if os.path.isfile(outMask) and usemask:
                        mask01 = cv2.imread(outMask, cv2.IMREAD_GRAYSCALE)
                        mask01 = (mask01 > 0).astype(np.uint8,copy=False)
                        if mask01.ndim == 3: mask01 = mask01[:,:,0]
                        mask_int = cv2.integral(mask01)
                        boxes = np.array(obj_px, dtype=int)  
                        x0, y0, x1, y1 = boxes.T
                        mask_sum = (mask_int[y1, x1] - 
                                        mask_int[y0, x1] - 
                                        mask_int[y1, x0] + 
                                        mask_int[y0, x0])
                        area_px = obj_w*obj_h
                        if np.sum(mask01)/sw/sh> overlap_photo:
                            tfx = np.zeros(mask_sum.shape[0], dtype=bool)      # No any invidual grains seleced if overall overlap is higher than threshold.
                        else:
                            tfx = mask_sum<=overlap_grain*area_px              # Ratio of mask2object.
                        del mask_sum

                        # Filterning data if existing.
                        obj_score = obj_score[tfx]
                        labels_r = labels_r[tfx,:]
                        area_r = area_r[tfx]
                        labels_yolo = labels_yolo[tfx,:]
                        obj_px = obj_px[tfx,:]
                        obj_score_output = obj_score_output[tfx]
                        
                    # Computing statistics before thresholding after segmentation.
                    nrn = len(obj_score)
                    area0 = sw*sh*resolution**2
                    pp0,zmax0,zmin0 = [zmax]*3
                    if nrn>0:
                        pp0 = np.mean(obj_score)*100
                        zmax0 = np.max(labels_r[:,5])
                        zmin0 = np.min(labels_r[:,3:5])
                        
                    # Selecting only data larger than thersold.
                    whr = {"w": 3, "h": 4, "r": 5}
                    cutmethod = PP.CutGrainSizeMethod
                    idcs = [whr[ch] for ch in cutmethod if ch in whr]
                    tf0 = obj_score >= conf1
                    for idc in idcs:
                        tf0 = tf0 & (labels_r[:,idc]<= szmax) & (labels_r[:,idc]>= szmin)
                        
                    labels_r = labels_r[tf0,:]
                    area_r = area_r[tf0]
                    labels_yolo_threshold = labels_yolo[tf0,:]
                    obj_px0 = obj_px[tf0,:].astype(int)
                    
                    nrs = labels_r.shape[0]                                    # Number of objects after thresholding.
            
                # Define characteristic percents and output variable name 1.
                pct = [0.1,0.5,0.6,0.84]                                       # D10, D50, D60, D84.
                cnames_m = []
                for weight_method in ['count','area']:
                    for size_method in ['x','y','r']:
                        for val in list(np.array(np.array(pct)*100,dtype='int')):
                            keys = "_".join(['D'+str(val),size_method,weight_method,'meter'])
                            cnames_m.append(keys)
                
                # Output variable names part 2: Mean_size_count_meter to Area_m2.
                idm1 = cnames_gsd.index('Mean_size_count_meter')
                idm2 = cnames_gsd.index('Area_m2')
                nv = idm2 - idm1 + 1
                cnames_b = cnames_gsd[idm1:idm1+nv]
                
                # Final output variable names and default values.
                cnames_merge = cnames_m + cnames_b
                dx_merge = np.ones(len(pct)*6 + nv)*zmax                       # Using default value if nrs<2
                
                # Computing metrics after thersholding.
                if nrs>=2:
                    zmc = np.mean(labels_r[:,5])
                    zma = np.sum(labels_r[:,5]*labels_r[:,6])/np.sum(labels_r[:,6])
                    pps = np.mean(labels_r[:,8])*100
                    rects_all = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in obj_px0]
                    area_all = unary_union(rects_all)
                    del rects_all 
                    area_ratio = area_all.area/sw/sh*100                       # Ratio of sediment to total area.
                    del area_all
                    obj_px02 = obj_px0[labels_r[:,5]<=0.002,:]                 # 0.002 is 2 mm.
                    area_ratio_2mm = 0.0
                    if obj_px02.shape[0]>0:
                        rects_2mm = [box(x1, y1, x2, y2) for x1, y1, x2, y2 in obj_px02]
                        area_2mm = unary_union(rects_2mm)
                        area_ratio_2mm = area_2mm.area/sw/sh*100               # Ratio of sediments less than 2mm to total area.
                        del rects_2mm, area_2mm
                    
                    dx_b = np.array([zmc,zma,nrs,pps,nrn,pp0,nra,pp00,zmax0,zmin0,area_ratio,area_ratio_2mm,conf1*100,area0])
                    
                    # Deleting data.
                    del obj_id, obj_x, obj_y, obj_w, obj_h, obj_r, obj_others
                    
                    # Computing GSD using width, height, and diagonal size.
                    fcw,xcw,_ = ecdfm(labels_r[:,3])
                    faw,xaw,_ = ecdfm(np.stack((labels_r[:,3],area_r),axis=1),'byarea')
                    fch,xch,_ = ecdfm(labels_r[:,4])
                    fah,xah,_ = ecdfm(np.stack((labels_r[:,4],area_r),axis=1),'byarea')
                    fcr,xcr,_ = ecdfm(labels_r[:,5])
                    far,xar,_ = ecdfm(np.stack((labels_r[:,5],area_r),axis=1),'byarea')
                    
                    # Computing characteristic GSD based on user-defined pct.
                    dwc = np.interp(pct,fcw,xcw)                               # Count and width based.
                    dwa = np.interp(pct,faw,xaw)                               # Area and width based
                    dhc = np.interp(pct,fch,xch)                               # Count and height based.
                    dha = np.interp(pct,fah,xah)                               # Area and height based
                    drc = np.interp(pct,fcr,xcr)                               # Count and diagonal based.
                    dra = np.interp(pct,far,xar)                               # Area and diagonal based
                    dx_m = np.hstack((dwc,dhc,drc,dwa,dha,dra))                # Total: 4*6.
                    
                    # Ouput information.
                    unit = 'px' if resolution <0 else 'cm'
                    sc = 1 if resolution <0 else 100
                    print(formatSpec12 %(nrs,obj_name,drc[0]*sc,drc[1]*sc,drc[2]*sc,drc[3]*sc,unit))
                    print(formatSpec13 %(nrs,obj_name,dra[0]*sc,dra[1]*sc,dra[2]*sc,dra[3]*sc,unit))
                    dx_merge = np.hstack((dx_m,dx_b))
                    del dx_m, dx_b

                # Final output information.
                dx_merge = pd.DataFrame([dx_merge], columns=cnames_merge)
                if notes == 'default':
                    scales_i = headers_sc.copy()
                    if fname in scales_all_copy["Name"].values:
                        #cnames_i = ["Make", "Model","Author","Keyword","Site_ID",
                        #            'Date_PST','Date_Second']
                        cnames_i = scales_all_copy.columns
                        matched = scales_all_copy.loc[
                            scales_all_copy["Name"] == fname, cnames_i
                        ].reset_index(drop=True)
                        scales_i.loc[0, cnames_i] = matched.loc[0, cnames_i]
                else:
                    scales_i = scales_all.iloc[[idx], :].reset_index(drop=True)
                dx_merge.index = scales_i.index
                dx_i = pd.concat([scales_i,dx_merge], axis=1)
                dx_i.loc[0, 'Name'] = fname
                dx_i.loc[0, 'Width_pixel'] = sw
                dx_i.loc[0, 'Height_pixel'] = sh
                dx_i.loc[0, 'L1L2_res_meter_per_pixel'] = resolution
                dx_i.loc[0, 'L1L2_res_millimeter_per_pixel'] = \
                    resolution * 1000.0 if resolution>0 else resolution
                dx_i.loc[0, 'Scale_x_meter'] = sw * abs(resolution)
                dx_i.loc[0, 'Scale_y_meter'] = sh * abs(resolution)
                dx_i.loc[0, 'Notes'] = notes
                dx_i.loc[0, 'Folder'] = foldername
                dx_i.loc[0, 'Grain_number_raw'] = nra
                dx_i.loc[0, 'Grain_number_segmentation'] = nrn
                dx_i.loc[0, 'Grain_number_threshold'] = nrs
                dx_i.loc[0, 'Accuracy_raw_percent'] = pp00
                dx_i.loc[0, 'Accuracy_segmentation_percent'] = pp0
                dx_i.loc[0, 'Accuracy_threshold_percent'] = pps
                dx_i.loc[0, 'Threshold_percent'] = conf1 * 100.0
                dx_i.loc[0, 'Area_m2'] = sw * sh * (resolution ** 2)
                dx_i.insert(2, 'Save_folder', os.path.basename(SaveDir))
                dx_i.insert(3, 'Model_ID', model_name)
                
                dtype_map = {
                    'Width_pixel':'int32','Height_pixel':'int32',
                    'L1L2_res_meter_per_pixel':pr,
                    'L1L2_res_millimeter_per_pixel':pr,
                    'Scale_x_meter':pr,'Scale_y_meter':pr,
                    'Grain_number_raw':'int32','Grain_number_segmentation':'int32',
                    'Grain_number_threshold':'int32',
                    'Accuracy_raw_percent':pr,
                    'Accuracy_segmentation_percent':pr,
                    'Accuracy_threshold_percent':pr,
                    'Threshold_percent':pr,'Area_m2':pr
                }
                dx_i = dx_i.astype({k:v for k,v in dtype_map.items() if k in dx_i.columns})
                
                # Adjusting the column names order for output.
                cnames_f = list(dx_i.columns)
                if PP.SaveStatisticsVersion.lower() == 'v1':
                    idx3a = cnames_f.index('Make')
                    idx3b = cnames_f.index('Notes')
                    cnames_save = cnames_f[0:idx3a] + cnames_f[idx3b+1:] + cnames_f[idx3a:idx3b+1]
                    
                elif PP.SaveStatisticsVersion.lower() == 'v3':
                    idx4a = cnames_f.index('D10_x_count_meter')
                    idx4b = cnames_f.index('Area_ratio_percent')
                    idx5a = cnames_f.index('Area_ratio_below_2mm_percent')
                    idx5b = cnames_f.index('Area_m2')
                    cnames_basics = ['Name','Folder','Save_folder','Scale_source','Model_ID','Width_pixel','Height_pixel',
                                      'L1L2_res_meter_per_pixel','L1L2_res_millimeter_per_pixel',
                                      'Scale_x_meter','Scale_y_meter','Relative_error_percent',
                                      'Photo_indicator']
                    cnames_save = cnames_basics + cnames_f[idx4a:idx4b+1] + cnames_f[idx5a:idx5b+1]
                else:
                    raise 'Invalid statistics output version'
                
                # Converting label data to dataframe.
                cnames_individual = ['label_index','center_width_m',\
                            'center_heigth_m','width_m','height_m','diagonal_m',\
                                'area_m2', 'resolution_m_per_pixel','probability']        
                if labels_r is not None:
                    labels_rr = pd.DataFrame(labels_r,columns=cnames_individual)
                    
                # Interpolating grain size data to uniform grid if needed.
                vtknames = [PP.SaveVTKName]
                if PP.SaveVTK and labels_r is not None:
                    nv = len(vtknames)
                    layers = np.zeros((nv,sh,sw), dtype=np.float32)
                    for nvi in range(nv):
                        vals = labels_rr[vtknames[nvi]].to_numpy(dtype=float)
                        for nrsi in range(nrs):
                            x1, y1, x2, y2 = obj_px0[nrsi, :]
                            layers[nvi, y1:y2, x1:x2] = vals[nrsi]
                            
                # Output 1: characterisitc grain size for photo i (csv).
                gsd_i =  dx_i[cnames_save]
                if PP.SaveStatisticsData:
                    gsd_i.to_csv(outLocalGSDName,index=False)
                    
                # Output 2: sizes of all individual grain sizes after thresholding (csv).
                if PP.SaveObjectData and labels_r is not None:
                    labels_rr.to_csv(outLocalSizeName,index=False)
                    
                #-------------------------------------------------------------#
                # Output 3,4,5: locations and sizes of all detected objects before#
                # and after thresholding (txt: 2 for before and 1 for after threshold).
                # Data before thresholding are saved to "BeforeThreshold". 2 extra
                # classes.txt files are also generated but only for one photo.
                if PP.SaveYOLOData:
                    # Output YOLO data before thresholding.
                    os.makedirs(outYOLOFolderBefore,exist_ok=True)
                    fmt0 = ('%d', '%8.6f', '%8.6f', '%8.6f', '%8.6f')
                    fmt1 = ('%d', '%8.6f')
                    delim = " "
                    np.savetxt(outYOLOBeforeTxt,labels_yolo,fmt = fmt0,delimiter=delim)
                    np.savetxt(outYOLOBeforeScoreTxt,obj_score_output,fmt = fmt1,delimiter=delim)
                    np.savetxt(outYOLOBeforeClassTxt,[obj_name],fmt="%s",delimiter=delim)
                    
                    # Output YOLO data after thresholding.
                    np.savetxt(outYOLOAfterTxt,labels_yolo_threshold,fmt = fmt0,delimiter=delim)
                    np.savetxt(outYOLOAfterClassTxt,[obj_name],fmt="%s",delimiter=delim)
            
                # Output 6, 7 ,8 ,9: overlay image + 3 GSD curves. 
                scsource = scales_i['Scale_source'].iloc[0].lower()
                fig_show = PP.PrintOnScreen
                useagg = False
                imgi = None
                dpi_overlay = PP.SaveOverlayResolution
                dot_r = PP.SaveOverlayScaleDotSize
                dot_c = PP.SaveOverlayScaleDotColor
                if PP.SaveFigure and PP.SaveOverlayLabel and nrs>= 2:
                    if not fig_show:
                        useagg = True
                        matplotlib.use("Agg")
                    from matplotlib import pyplot as plt
                    plt.rcParams.update({
                        'font.family': 'serif',
                        'font.serif': ['Times New Roman'],
                        'mathtext.fontset': 'stix',
                        'pdf.fonttype': 42,
                        'svg.fonttype': 'none',
                        'axes.unicode_minus': False,
                    })
                    outFileName = SaveDir + os.sep + stem + '_'  + outkeyword + '_box.jpg'
                    imgi = Image.open(imgpath).convert('RGB')
                    imgi = np.array(imgi)                                      # Converting image to array format.
                    imgi = cv2.cvtColor(imgi, cv2.COLOR_RGB2BGR)               # Converting image from rgb to bgr.
                    
                    # For sediment overlay.
                    # Color: green (0,255,0); blue (0,0,255); red (255,0,0); whilte (255,255,255)
                    lc1_bgr = tuple(reversed(line_color1_rgb))                 # Convert to bgr.
                    plt.figure(1)
                    for bbox in obj_px0:                                       # For sediments.
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(imgi, (x1, y1), (x2, y2),lc1_bgr,llw)

                    # For scale overlay. 
                    lc2_bgr = tuple(reversed(line_color2_rgb))         # In bgr.
                    L1 = ['L1_px1_pixel','L1_py1_pixel','L1_px2_pixel','L1_py2_pixel']
                    L2 = ['L2_px1_pixel','L2_py1_pixel','L2_px2_pixel','L2_py2_pixel']
                    box12 = scales_i[L1+L2].to_numpy().astype(int)
                    box12 = box12[box12>0]
                    if len(box12)>0:
                        obj_pxs = box12.reshape((4,-1),order='F').T            # Scale bounding box.
                        for bbox in obj_pxs:                                   # For scales.
                            x1, y1, x2, y2 = bbox
                            if scsource == 'line':
                                cv2.line(imgi, (x1, y1), (x2, y2), lc2_bgr, llw+2)
                                cv2.circle(imgi, (x1, y1), dot_r, dot_c, -1)
                                cv2.circle(imgi, (x2, y2), dot_r, dot_c, -1)
                            else:
                                cv2.rectangle(imgi, (x1, y1), (x2, y2), lc2_bgr, llw+2)
                                if len(mask01)>0: mask01[y1:y2,x1:x2] = 0      # Removing scale from mask.
                        del obj_pxs
                        
                    # Plotting overlay for segmented regions.
                    if len(mask01)>0:
                        lc3_bgr = tuple(reversed(line_color3_rgb))             # In bgr.
                        imgi[mask01 ==1] = ((1-segalpha) * np.array(lc3_bgr, dtype=np.uint8) + 
                                             segalpha * imgi[mask01 == 1] ).astype(np.uint8)
                        if not PP.SaveOverlayAll: del mask01
                                                
                    plt.imshow(cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB))
                    plt.axis("off")
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.savefig(outFileName, bbox_inches='tight',pad_inches=0, dpi=dpi_overlay)
                    plt.show() if fig_show else plt.close(1)
                    plt.close('all')
                    imgi = None
            
                if PP.SaveFigure and PP.SaveGSDCurve and nrs>= 2:
                    if not useagg and not fig_show: 
                        matplotlib.use("Agg")
                    from matplotlib import pyplot as plt
                    plt.rcParams.update({
                        'font.family': 'serif',
                        'font.serif': ['Times New Roman'],
                        'mathtext.fontset': 'stix',
                        'pdf.fonttype': 42,
                        'svg.fonttype': 'none',
                        'axes.unicode_minus': False,
                    })
                    dpi_gsd = PP.SaveGSDCurveResolution
                    
                    #% Grain size distribution: using width data.
                    outFigWidth = SaveDir + os.sep + stem + '_' + outkeyword + '_Width' + '.jpg'
                    plt.figure(2,figsize=(5,3))
                    plt.plot(xcw*sc,fcw*100, color='red', linewidth=1,marker='o',markersize=3,label='Width: by count')
                    plt.plot(xaw*sc,faw*100, color='black', linewidth=1,marker='s',markersize=3,label='Width: by area')
                    xmin = xaw[0]*sc; xmax = xaw[-1]*sc; xmax_p = xmin + (xmax - xmin)*0.8
                    plt.plot(np.array([xmin,xmax_p]),np.array([50,50]), linestyle='--', color='black', linewidth=1)
                    plt.legend()
                    plt.xlim(xaw[0]*sc, xaw[-1]*sc)
                    plt.ylim(0, 100)
                    plt.xlabel(f'Grain Size ({unit})')
                    plt.ylabel('Percent finer (%)')
                    plt.grid(True)
                    plt.savefig(outFigWidth, bbox_inches='tight',dpi=dpi_gsd)
                    plt.show() if fig_show else plt.close(2)
            
                    #% Grain size distribution: using height data.
                    outFigHeight = SaveDir + os.sep + stem + '_' + outkeyword + '_Height' + '.jpg'
                    plt.figure(3,figsize=(5,3))
                    plt.plot(xch*sc,fch*100, color='red', linewidth=1,marker='o',markersize=3,label='Height: by count')
                    plt.plot(xah*sc,fah*100, color='black', linewidth=1,marker='s',markersize=3,label='Height: by area')
                    xmin = xah[0]*sc; xmax = xah[-1]*sc; xmax_p = xmin + (xmax - xmin)*0.8
                    plt.plot(np.array([xmin,xmax_p]),np.array([50,50]), linestyle='--', color='black', linewidth=1)
                    plt.legend()
                    plt.xlim(xah[0]*sc, xah[-1]*sc)
                    plt.ylim(0, 100)
                    plt.xlabel(f'Grain Size ({unit})')
                    plt.ylabel('Percent finer (%)')
                    plt.grid(True)
                    plt.savefig(outFigHeight, bbox_inches='tight',dpi=dpi_gsd)
                    plt.show() if fig_show else plt.close(3)
            
                    #% Grain size distribution: using Diagonal distance.
                    outFigDiag = SaveDir + os.sep + stem + '_' + outkeyword + '_Diagonal' + '.jpg'
                    plt.figure(4,figsize=(5,3))
                    plt.plot(xcr*sc,fcr*100, color='red', linewidth=1,marker='o',markersize=3,label='Diagonal: by count')
                    plt.plot(xar*sc,far*100, color='black', linewidth=1,marker='s',markersize=3,label='Diagonal: by area')
                    xmin = xar[0]*sc; xmax = xar[-1]*sc; xmax_p = xmin + (xmax - xmin)*0.8
                    plt.plot(np.array([xmin,xmax_p]),np.array([50,50]), linestyle='--', color='black', linewidth=1)
                    plt.legend()
                    plt.xlim(xar[0]*sc, xar[-1]*sc)
                    plt.ylim(0, 100)
                    plt.xlabel(f'Grain Size ({unit})')
                    plt.ylabel('Percent finer (%)')
                    plt.grid(True)
                    plt.savefig(outFigDiag, bbox_inches='tight',dpi=dpi_gsd)
                    plt.show() if fig_show else plt.close(4)
                    plt.close('all')
                
                if PP.SaveFigure and PP.SaveOverlayAll and nrs>= 2:
                    if not useagg and not fig_show: 
                        matplotlib.use("Agg")
                    from matplotlib import pyplot as plt
                    import matplotlib.patheffects as pe
                    plt.rcParams.update({
                        'font.family': 'serif',
                        'font.serif': ['Times New Roman'],
                        'mathtext.fontset': 'stix',
                        'pdf.fonttype': 42,
                        'svg.fonttype': 'none',
                        'axes.unicode_minus': False,
                    })
                    swh = min(sw,sh)
                    conditions = [(swh <= 500),
                                  (swh <= 1000),
                                  (swh <= 1500),
                                  (swh <= 2000),
                                  (swh <= 2500),
                                  (swh <= 3000),
                                  (swh <= 3500),
                                  (swh <= 4000),
                                  (swh <= 4500)]
                    choices = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,5]
                    beta = np.select(conditions, choices, default=5.5)
                    beta = PP.OverlayFontSizeScaleFactor
                    
                    # User-specified visualization control.
                    slw = PP.OverlayFontLineWidth*beta
                    fz = PP.OverlayFontSize*beta
                    lfz = PP.OverlayLegendFontSize*beta
                    gsdlw = PP.OverlayGSDLineWidth * beta
                    gsdticklw = PP.OverlayGSDTickLineWidth*beta
                    ms = PP.OverlayD50MarkerSize*beta
                    ml = PP.OverlayGSDTickMarkLength*beta
                    ftc = PP.OverlayFontColor
                    fftc = PP.OverlayFontColorForeground
                    pad_inches = 0.01
                    
                    # Prepare data.
                    xc = [xcw, xch, xcr, xaw, xah, xar]
                    fc = [fcw, fch, fcr, faw, fah, far]
                    del fcw, xcw, faw, xaw, fch, xch, fah, xah, fcr, xcr, far, xar
                    colors = PP.OverlayGSDColor
                    colorsm = PP.OverlayGSDManualColor
                    colorsf = PP.OverlayGSDFieldColor
                    ls = PP.OverlayGSDLineType
                    keys = PP.OverlayGSDKey
                    lgs = PP.OverlayGSDLegend
                    dmin = PP.OverlayGSDMinSize
                    dmax = PP.OverlayGSDMaxSize
                    
                    # Loading manual grain size label data if existed..
                    csvname = stem + '.txt'
                    fkeyword = PP.FieldDataFolderName + '_' + \
                        PP.FieldDataConversionFolderName
                    csvfile = os.path.join(WorkDir,PP.ManualGrainFolderName,csvname)
                    xcm = []
                    if os.path.isfile(csvfile) and PP.SaveOverlayManual:
                        lbs = np.loadtxt(csvfile)                              # Loading manual data.
                        whra = np.empty((lbs.shape[0],4))
                        whra[:,0] =  lbs[:,3]*sw*abs(resolution)
                        whra[:,1] =  lbs[:,4]*sh*abs(resolution)
                        whra[:,2] = np.sqrt(whra[:,0]**2 + whra[:,1] **2)
                        whra[:,3] = whra[:,0] * whra[:,1] 
                        nrm = whra.shape[0]
                        fcw,xcw,_ = ecdfm(whra[:,0])
                        faw,xaw,_ = ecdfm(whra[:,[0,3]],'byarea')
                        fch,xch,_ = ecdfm(whra[:,1])
                        fah,xah,_ = ecdfm(whra[:,[1,3]],'byarea')
                        fcr,xcr,_ = ecdfm(whra[:,2])
                        far,xar,_ = ecdfm(whra[:,[2,3]],'byarea')
                        xcm = [xcw, xch, xcr, xaw, xah, xar]
                        fcm = [fcw, fch, fcr, faw, fah, far]
                        del lbs, whra
                        del fcw, xcw, faw, xaw, fch, xch, fah, xah, fcr, xcr, far, xar

                    # Identify which data to visualize.
                    gwt = "".join(dict.fromkeys(PP.GSDWeight.lower()))
                    gax = "".join(dict.fromkeys(PP.GSDAxis.lower()))
                    ovflag0 = gwt + gax
                    ovflag = '_' + ovflag0 if ovflag0 != 'ar' else ''
                    
                    tags = ['wc','hc','rc','wa','ha','ra']
                    tag_id = {tag: i for i, tag in enumerate(tags)}
                    idx = [tag_id[p] for 
                           p in (a + w for a in gax for w in gwt) if p in tag_id]
                    idx = np.sort(idx)
                    outovname = stem + '_' + outkeyword + ovflag + '.jpg'
                    outFigOverlayGSD = SaveDir + os.sep + outovname 
                    
                    # Obtaning the range of x and f.
                    xmin, xmax, ymin, ymax = [0]*4
                    for i in idx:
                        xci = xc[i]; fci = fc[i]
                        if xcm:
                            xci = np.concatenate((xci, xcm[i]), axis=0)
                            fci = np.concatenate((fci, fcm[i]), axis=0)           
                        x = np.asarray(xci, dtype=float) * sc
                        f = np.asarray(fci, dtype=float) * 100.0  
                        xmin = min(xmin,x.min())
                        xmax = max(xmax,x.max())
                        ymin = min(ymin,f.min())
                        ymax = max(ymax,f.max())
                        del xci, fci
                    xmin = 0 if dmin is None else dmin 
                    xmax0 = np.ceil(xmax) if dmax is None else dmax*100
                    ymin = 0; ymax0 = 100
                    
                    img_aspect = sw / sh
                    xr = xmax0 - xmin
                    yr = ymax0 - ymin
                    data_aspect = xr / yr
                    xrn = xr; yrn = yr
                    if data_aspect < img_aspect:
                        xrn = yr * img_aspect
                        xmax = xmin + xrn
                    else:
                        yrn = xr / img_aspect
                        ymax = ymin + yrn
                    sx = xrn/xr
                    sy = yrn/yr
                    
                    ###########################################################
                    if imgi is None:
                        imgi = Image.open(imgpath).convert('RGB')
                        imgi = np.array(imgi)                                  # Converting image to array format.
                        imgi = cv2.cvtColor(imgi, cv2.COLOR_RGB2BGR)           # Converting image from rgb to bgr.
                        
                    # For sediment overlay.
                    # Color: green (0,255,0); blue (0,0,255); red (255,0,0); whilte (255,255,255)
                    lc1_bgr = tuple(reversed(line_color1_rgb))                 # Convert to bgr.
                    if PP.SaveOverlayGrain:
                        for bbox in obj_px0:                                   # For sediments.
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(imgi, (x1, y1), (x2, y2),lc1_bgr,llw)
                            
                    # For scale overlay. 
                    lc2_bgr = tuple(reversed(line_color2_rgb))         # In bgr.
                    L1 = ['L1_px1_pixel','L1_py1_pixel','L1_px2_pixel','L1_py2_pixel']
                    L2 = ['L2_px1_pixel','L2_py1_pixel','L2_px2_pixel','L2_py2_pixel']
                    box12 = scales_i[L1+L2].to_numpy().astype(int)
                    box12 = box12[box12>0]
                    if PP.SaveOverlayScale:
                        if len(box12)>0:
                            obj_pxs = box12.reshape((4,-1),order='F').T        # Scale bounding box.
                            for bbox in obj_pxs:                               # For scales.
                                x1, y1, x2, y2 = bbox
                                if scsource == 'line':
                                    cv2.line(imgi, (x1, y1), (x2, y2), lc2_bgr, llw+2)
                                    cv2.circle(imgi, (x1, y1), dot_r, dot_c, -1)
                                    cv2.circle(imgi, (x2, y2), dot_r, dot_c, -1)
                                else:
                                    cv2.rectangle(imgi, (x1, y1), (x2, y2), lc2_bgr, llw+2)
                                    if len(mask01)>0: mask01[y1:y2,x1:x2] = 0  # Removing scale from mask.
                            del obj_pxs
                            
                    # Plotting overlay for segmented regions.
                    lc3_bgr = tuple(reversed(line_color3_rgb))                 # In bgr.
                    if PP.SaveOverlaySegmentation:
                        if len(mask01)>0:
                            imgi[mask01 ==1] = ((1-segalpha) * np.array(lc3_bgr, dtype=np.uint8) + 
                                                 segalpha * imgi[mask01 == 1] ).astype(np.uint8)
                            del mask01
                    ###########################################################
                    
                    # Create figure.
                    #fig, ax = plt.subplots(num=5, clear=True,figsize=(fig_w, fig_h),dpi=dpi_overlay)   
                    fig, ax = plt.subplots(num=5, clear=True)   
                    if PP.SaveOverlayGSD:
                        if PP.SaveOverlayImage:
                            ax.imshow(cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB), 
                                      extent=[xmin, xmax, ymin, ymax], 
                                      origin="upper", aspect="equal")
                    else:
                        if PP.SaveOverlayImage:
                            ax.imshow(cv2.cvtColor(imgi, cv2.COLOR_BGR2RGB), 
                                      origin="upper",aspect="equal")
                            ax.axis("off")
                            
                    if PP.SaveVTK:
                        usernames = [PP.SaveVTKNameShow]
                        lid = 0                                               
                        layer = layers[lid, :, :].copy()
                        del layers
                        layer[layer==0] =  np.nan
                        layer = np.log2(layer*1000)
                        vname = vtknames[lid]
                        if usernames: vname = usernames[lid]                   # Using User-specified names.
                        #vmin = np.floor(np.nanmin(layer))
                        #vmax = np.ceil(np.nanmax(layer))
                        vmin = np.nanmin(layer)
                        vmax = np.nanmax(layer)
                        #ratio = PP.SaveVTKRatio
                        #swn, shn = int(sw*ratio), int(sh*ratio)
                        #layer = cv2.resize(layer, (swn, shn), interpolation=cv2.INTER_AREA)
                        if not PP.SaveOverlayGSD and PP.SaveOverlayImage:
                            extent = [0, sw, sh, 0]
                        else:
                            extent = [xmin, xmax, ymin, ymax]
                        im = ax.imshow(
                            layer,
                            extent=extent,
                            origin="upper",
                            cmap=PP.SaveVTKCmap,
                            alpha=PP.SaveVTKAlpha,
                            aspect="equal",
                            interpolation='none',
                            vmin=vmin,
                            vmax=vmax
                        )
                        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.005, fraction=0.02)
                        ax_pos = ax.get_position()
                        cbar.ax.set_position([
                            ax_pos.x1 + 0.005,                                 # x-position (slightly to the right)
                            ax_pos.y0,                                         # y-position (same bottom)
                            0.02,                                              # colorbar width
                            ax_pos.height                                      # same height as image
                        ])
                        #cbar.set_label(f"${vname}$", fontsize=PP.SaveVTKFontSize)
                        cbar.set_label(f"$\\mathrm{{{vname}}}$", fontsize=PP.SaveVTKFontSize)
                        del layer
                        
                    if PP.SaveOverlayGSD:
                        for i in idx:                                      
                            x = np.asarray(xc[i], dtype=float) * sc
                            f = np.asarray(fc[i], dtype=float) * 100.0  
                            order = np.argsort(f)
                            x_sorted = x[order]
                            f_sorted = np.clip(f[order], ymin, ymax)
                            
                            # Plot gsd curve.
                            ni = x_sorted.shape[0]
                            labeli = keys[0]+lgs[i] + f' (n = {nrs:,})'
                            ax.plot(x_sorted*sx, f_sorted*sy, 
                                    linestyle=ls[0], lw=gsdlw, color=colors[i],label=labeli)
                            x00 = float(np.interp(50.0, f_sorted, x_sorted))
                            ax.plot(x00*sx, 50.0*sy, "o", color='red', markersize=ms, 
                                    markeredgecolor="k",markeredgewidth=0.5)
                            
                            # Plot for manual label data.
                            if xcm:
                                x = np.asarray(xcm[i], dtype=float) * sc
                                f = np.asarray(fcm[i], dtype=float) * 100.0  
                                order = np.argsort(f)
                                x_sorted = x[order]
                                f_sorted = np.clip(f[order], ymin, ymax)
                                ni = x_sorted.shape[0]
                                labeli = keys[1]+lgs[i] + f' (n = {nrm:,})'
                                
                                # Plot gsd curve.
                                ax.plot(x_sorted*sx, f_sorted*sy, 
                                        linestyle=ls[1], lw=gsdlw, color=colorsm[i],label=labeli)
                                x0 = float(np.interp(50.0, f_sorted, x_sorted))
                                ax.plot(x0*sx, 50.0*sy, "o", color='red', markersize=ms, 
                                        markeredgecolor="k",markeredgewidth=0.5)
                        
                    # Loading field data if exists.
                    fkeyword = PP.FieldDataFolderName + '_' + \
                        PP.FieldDataConversionFolderName
                    folder = os.path.join(WorkDir, fkeyword)
                    fieldfiles = []
                    primary = os.path.join(folder, f"{stem}.csv")
                    if os.path.isfile(primary): fieldfiles.append(primary)
                    matches = glob.glob(os.path.join(folder, f"{stem}_*.csv"))
                    matches.sort(key=os.path.getmtime)
                    fieldfiles.extend(matches)
                    
                    # linestyle for field data.
                    nff = min(len(colors),len(fieldfiles))
                    lgsfu = PP.OverlayGSDLegendField
                    if nff>0 and PP.SaveOverlayField:
                        for k in range(nff):
                            fieldfile = fieldfiles[k]
                            fieldname = os.path.basename(fieldfile).replace('.csv','')
                            fieldname = fieldname.split('_')
                            axk = fieldname[1]
                            unitk = fieldname[2]
                            lgsf = ''
                            if lgsfu is None:
                                if len(fieldname) > 3: lgsf = ', ' + ' '.join(fieldname[3:])
                            else:
                                if k < len(lgsfu): lgsf = lgsfu[k] 
                                
                            xfm = pd.read_csv(fieldfile)
                            ni = int(xfm['grain_number'].iloc[0])
                            cnames = xfm.columns
                            xfm = xfm.to_numpy()
                            if unit == 'px' and unitk == 'meters':
                                print('Resolution is required to convert field data from meters to pixels')
                            else:
                                if unit == 'cm' and unitk == 'pixels': sc = sc*resolution
                                xfm[:,1] = xfm[:,1]*sc                         # Grain size.
                                if axk == 'b' and ('w' in gax or 'h' in gax):
                                    if lgsfu is None:
                                        labeli = keys[2] + 'count-b' + lgsf + f' (n = {ni:,})'
                                    else:
                                        labeli = keys[2] + lgsf + f' (n = {ni:,})'
                                    ax.plot(xfm[:,1]*sx, xfm[:,2]*sy,linestyle=ls[2],
                                            lw=gsdlw,color=colorsf[k],label=labeli)
                                    x1 = float(np.interp(50.0, xfm[:,2], xfm[:,1]))
                                    ax.plot(x1*sx, 50.0*sy, "o", color='red', 
                                            markersize=ms, markeredgecolor="b",markeredgewidth=0.3)

                                    if xfm.shape[1] == 4 and ('a' in gwt):
                                        if lgsfu is None:
                                            labeli = keys[2] + 'area-b' + lgsf + f' (n = {ni:,})'
                                        else:
                                            labeli = keys[2] + lgsf + f' (n = {ni:,})'
                                        ax.plot(xfm[:,1]*sx, xfm[:,3]*sy,
                                                marker='+', markersize=2,
                                                markerfacecolor='white',
                                                markeredgewidth=0.5,        
                                                linestyle=ls[3], 
                                                lw=gsdlw,
                                                color=colorsf[k],label=labeli)   
                                        x2 = float(np.interp(50.0, xfm[:,3], xfm[:,1]))
                                        ax.plot(x2*sx, 50.0*sy, "o", color='red', 
                                                markersize=ms, markeredgecolor="b",markeredgewidth=0.3)
                                        
                    # Adjust the overal xlabel and ylabel.
                    if PP.SaveOverlayGSD or PP.SaveOverlayField:
                        nx = PP.OverlayGSDXTickNumber
                        ny = PP.OverlayGSDYTickNumber
                        ax.set_xlim(xmin, xmax)
                        ax.set_ylim(ymin, ymax)
                        yticks = np.linspace(ymin, ymax, ny)          
                        ax.set_yticks(yticks)
                        ax.tick_params(axis='x', labelsize=gsdticklw, length=ml)
                        ax.tick_params(axis='y', labelsize=gsdticklw, length=ml)
                        if sx != 1:
                            xticks = np.linspace(0, xrn, nx)          
                            xticklabels = np.linspace(0, xr, nx)    
                            ax.set_xticks(xticks)
                            ax.set_xticklabels([f"{lab:.0f}" for lab in xticklabels])  
                        else:
                            yticks = np.linspace(0, yrn, ny)          
                            yticklabels = np.linspace(0, yr, ny)    
                            ax.set_yticks(yticks)
                            ax.set_yticklabels([f"{lab:.0f}" for lab in yticklabels])  
                        
                        fdistx = PP.OverlayFontBoundaryDistanceX
                        fdisty = PP.OverlayFontBoundaryDistanceY
                        if PP.SaveOverlayShowFont:
                            ax.text((xmin + xmax) / 2.0,
                                    ymin + fdistx * (ymax - ymin),
                                    f"Grain size ({unit})", ha="center", va="bottom",
                                    fontsize=fz, color=ftc,
                                    path_effects=[pe.withStroke(linewidth=slw, foreground=fftc)])
                            ax.text(xmin + fdisty * (xmax - xmin),
                                    (ymin + ymax) / 2.0,
                                    "Percent finer (%)", ha="left", va="center", rotation=90,
                                    fontsize=fz, color=ftc,
                                    path_effects=[pe.withStroke(linewidth=slw, foreground=fftc)])
                        
                        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        
                        # Control visualization of axis and grid.
                        showax = False
                        if not PP.SaveOverlayImage:
                            showax = True
                            ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
                        for spine in ax.spines.values(): spine.set_visible(showax)
                            
                        if PP.SaveOverlayGSDLegend:
                            xshift =  PP.OverlayLegendShiftX
                            yshift =  PP.OverlayLegendShiftY
                            sf = None if (xshift is None or yshift is None) else (xshift, yshift)
                            ax.legend(loc=PP.OverlayLegendLoc,
                                      bbox_to_anchor=sf,
                                      bbox_transform=ax.transAxes if sf else None,
                                      fontsize=lfz,
                                      frameon=PP.OverlayLegendFrame,           # draw frame
                                      fancybox=True,                           # rounded corners
                                      shadow=PP.OverlayLegendShadow,           # soft shadow
                                      facecolor=PP.OverlayLegendFaceColor,
                                      edgecolor=PP.OverlayLegendEdgeColor, 
                                      framealpha=PP.OverlayLegendAlpha,        # transparency
                                      borderpad=0.03,                          # padding inside box
                                      labelspacing=0.05,                       # vertical spacing between labels
                                      handlelength=1.4,                        # line length in legend
                                      handleheight = 0.9,
                                      handletextpad=0.1,                       # gap between handle and text
                                      alignment=PP.OverlayLegendAlignment,
                                      labelcolor=PP.OverlayLegendColor
                                      )

                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.savefig(outFigOverlayGSD, dpi=dpi_overlay, bbox_inches="tight", pad_inches=pad_inches)
                    plt.show() if fig_show else plt.close(5)    
                    del imgi
                    
                # Printing information.
                dt = time.time() - ts
                
            # Adding the new gsd_i into the gsd for all photo.
            count = count + nrs
            if PP.SaveStatisticsData:                                          # Write data during loop.
                gsd_i.to_csv(outGSDName, mode="a", header=first_write, index=False)
                first_write = False
            else:
                rows.append(gsd_i)
                
        # Final satistics data.
        gsd = pd.read_csv(outGSDName, keep_default_na=False) if PP.SaveStatisticsData else (
              pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
        print(formatSpec14 %(outGSDName))
        print(formatSpec15 %(count, count/len(gsd)))
        
    t_end = time.time()
    tt = t_end - t_start
    print(formatSpec16 %(tt,tt/nf))
    #print(formatSpec17)
    
    return gsd

#%% Run the Photo2GSDSingle multiple times for multiple models.
#@profile
def Photo2GSD(PP):
    
    # Printing information control.
    formatSpec0 = 65*'='
    formatSpec1 = 'AI4GSD: an AI and cloud powered tool for grain size quantification'
    formatSpec2 = 'AI models in use: %s'
    formatSpec3 = 'Confidence thresholds in use: %s\n'
    formatSpec4 = 'Summary statistics file "%s" existed in %s, skipping...'
    formatSpec5 = 65*'-'
    formatSpec6 = ''
    formatSpec7 = 'Output all summary data to: %s'
    formatSpec8 = 'Total execution time: %3.2f s'
    
    # Checking if working diretory has been specified.
    if PP.Directory == '': raise ValueError('Working directory is not defined.')
    
    # Identifying if using single model or multiple model modes.
    if type(PP.ModelName) == str: 
        modelnames = list([PP.ModelName])
    elif type(PP.ModelName) == list:
        modelnames = PP.ModelName
    else:
        raise 'Invalid model name inputs.'
    
    # Update model name to x.0.0 format if x does not include 0.0.
    modelnames = [m + ".0.0" if len(m.split(".")) == 1 else m for m in modelnames]
    
    # Identifying if confidence threshold is a single or multiple values.
    if type(PP.ConfidenceThresholdUser) == float or type(PP.ConfidenceThresholdUser) == int: 
        thresholds = list([PP.ConfidenceThresholdUser])
    elif type(PP.ConfidenceThresholdUser) == list:
        thresholds = PP.ConfidenceThresholdUser
    else:
        raise 'Invalid confidence threshold inputs.'
    
    # Print information on screen.
    print(formatSpec0)
    print(formatSpec1)
    print(formatSpec2 %(', '.join(modelnames)))
    print(formatSpec3 %(', '.join(map(str,thresholds))))
    t_start = time.time()
    flag = '_'.join(modelnames) + '_' + "_".join(f"{thres*100:.0f}" for thres in thresholds)
    flag += '_' + PP.GSDFlag if PP.GSDFlag else ''
    
    # Pass parameters from PP to local.
    OverWriteGSDFileSummary = PP.OverWriteGSDFileSummary
    OverWrtieAll = PP.OverWriteAll
    GSDNameStem,_ = PP.StatisticsTemplateName.split('.')
    outGSDName = PP.Directory + os.sep + GSDNameStem + '_' + \
        os.path.basename(PP.Directory) + '_' + flag + '.csv'
        
    # Skip multimodel files if already existed.
    gsd_all = pd.DataFrame()
    if os.path.isfile(outGSDName) and (not OverWriteGSDFileSummary) and not OverWrtieAll:
        print(formatSpec4 %(os.path.basename(outGSDName), os.path.basename(PP.Directory)))
        gsd_all = pd.read_csv(outGSDName,keep_default_na=False)
    else:
        # Compouting results.
        gsd_list = []; 
        PPC = copy.deepcopy(PP)
        for thres in thresholds:
            PPC.ConfidenceThresholdUser = thres
            for model in modelnames:
                print(formatSpec5)
                PPC.ModelName = model
                gsd = Photo2GSDSingle(PPC)
                gsd_list.append(gsd)
                print(formatSpec6)
        gsd_all = pd.concat(gsd_list, ignore_index=True)
        
        # Output data.
        if PPC.SaveStatisticsData and len(gsd_all)>0:
            gsd_all.to_csv(outGSDName,index=False)
            
    t_end = time.time()
    print(formatSpec7 %(outGSDName))
    print(formatSpec8 %(t_end - t_start))
    print(formatSpec0)  
     
    return gsd_all

#%%