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
import os, re, gzip, shutil, platform, tarfile, glob, math, zipfile
import numpy as np
import pandas as pd
from natsort import natsorted, natsort_keygen
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
from exiftool import ExifToolHelper

from .mathm import dms2degree, numCount, cumtrapz, getFolders
from .mathm import __file__ as mathm_path

#%% This function reads filename into strings, applicable for all files.      %
def readOF(filename, keepWhiteSpace=True, ext=".gz"):
    
    # Checking if file is .gz file; unzip it if yes.
    if not os.path.isfile(filename):
        print("File not existed, checking if compressed file exists.")
        filename_new = filename + ext
        if filename_new.exists():
            print("Compressed file exists. Decompressing...")
            with gzip.open(filename_new, "rb") as fin, open(filename, "wb") as fout:
                shutil.copyfileobj(fin, fout)
                
    # Print error information if file not found.
    if not os.path.isfile(filename): raise FileNotFoundError(f"readOF: file not found: {filename}")
    
    # Read file if no error.
    p = Path(filename)
    lines = p.read_text(encoding="utf-8", errors="ignore")
    lines = re.sub(r'(?m)^\s*\r?\n', '', lines)                                # Remove empty or white-space only lines. 
    lines = lines.splitlines(True)
    
    # Converting list to array format.
    if keepWhiteSpace: 
        lines = np.array([ln.rstrip("\r\n") for ln in lines], dtype=object)
    else:
        lines = np.array([ln.strip() for ln in lines], dtype=object)

    return lines

#%% This function reads drone SRT log file to csv and array format.           %
def readSRT(VPath, FCPath = ''):    
    
    # Extract the parent folder path, file name stem.
    filepath = str(Path(VPath).parent)
    name = Path(VPath).stem
    SRTPath = os.path.join(filepath, name + '.SRT')
    OutPath = os.path.join(filepath, name + '_GPS.csv')

    # Read the whole file as txt.
    strs = readOF(SRTPath)                                                   
    strs = strs.reshape(6, -1, order='F').T                                
    nf = strs.shape[0]
    
    # Frame index
    index = " ".join(strs[:, 0].tolist())
    index = np.fromstring(index, sep=" ", dtype=int)
    index = index.reshape(nf,1)
    
    # Start/End time
    ti = " ".join(strs[:, 1].tolist()).replace("-->", " ").replace(":", " ").replace(",", " ")
    ti = np.fromstring(ti, sep=" ", dtype=float)
    time = ti.reshape(-1, nf,order='F').T

    # Date line -> [Y M D h m s ms us]
    dates = " ".join(strs[:, 3].tolist()).replace(":", " ").replace(",", " ").replace("-", " ")
    dates = np.fromstring(dates, sep=" ", dtype=float)
    dates = dates.reshape(-1, nf, order='F').T
    
    # Optional FC alignment
    if os.path.isfile(FCPath):
        fc = pd.read_csv(FCPath)
        ref0 = fc.iloc[0, 1:7].to_numpy()                                      # [Y M D h m s]

        base_df = pd.DataFrame(dates[:, :6], columns=["Y", "M", "D", "h", "m", "s"]).astype(int)
        base_dt = pd.to_datetime(base_df.rename(columns={
            "Y": "year", "M": "month", "D": "day", "h": "hour", "m": "minute", "s": "second"
        }))
        ref_dt = pd.Timestamp(datetime(int(ref0[0]), int(ref0[1]), int(ref0[2]),
                                       int(ref0[3]), int(ref0[4]), int(ref0[5])))
        etimes = (base_dt - ref_dt).dt.total_seconds().to_numpy()
        ts = etimes + dates[:, 6] / 1e3 + dates[:, 7] / 1e6

        if ("isVideo" in fc.columns) and ("time(second)" in fc.columns):
            cond = (fc["isVideo"].to_numpy() == 1) & (np.abs(fc["time(second)"].to_numpy() - ts[0]) < 30)
            idx = np.flatnonzero(cond)
            if idx.size:
                idv = idx[0]
                dt = float(fc.loc[idv, "time(second)"]) - ts[0]
                tsn = ts + dt
                tsn_s = np.floor(tsn).astype(int)
                subs = tsn - tsn_s
                aligned = (ref_dt + pd.to_timedelta(tsn_s, unit="s"))
                datevec6 = np.column_stack([
                    aligned.year,
                    aligned.month,
                    aligned.day,
                    aligned.hour,
                    aligned.minute,
                    aligned.second,
                ]).astype(float)            
                ms = np.floor(subs * 1e3).astype(float)
                us = np.round(subs * 1e6 - ms * 1e3).astype(float)
                dates = np.column_stack([datevec6, ms, us])

    # GPS (from column 5) -> pick fields 8,9,10 = latitude,longitude,altitude
    gps = " ".join(strs[:, 4].tolist()).replace("[", " ")
    gps = gps.split("]")
    gps = np.array(gps[:-1]).reshape(-1,nf,order='F').T

    xyz = gps[:, 7:10]
    xyz = " ".join(xyz.reshape(-1).tolist())
    xyz = (xyz.replace(":", " ").
           replace("latitude", " ").
           replace("longitude", " ").
           replace("altitude", " ").
           strip())
    xyz = np.fromstring(xyz, sep=" ", dtype=float)
    xyz = xyz.reshape(3,nf,order='F').T

    # Assemble data to dataframe.
    data = np.column_stack([index.astype(float), time, dates, xyz])
    names = [
        "Frame",
        "Start_H","Start_M","Start_S","Start_MS",
        "End_H","End_M","End_S","End_MS",
        "Year","Month","Day","Hour",
        "Minute","Second","MilliSecond","MicroSecond",
        "Latitude","Longitude","Altitude"
    ]
    F = pd.DataFrame(data, columns=names).apply(pd.to_numeric, errors="coerce")

    Path(OutPath).unlink(missing_ok=True)
    F.to_csv(OutPath, index=False)
    
    return F, data

#%% Extracting the EXIF information of a photo.
def readEXIFPhoto(img_path: str, zmax=-9999) -> dict:
    
    # Make sure file extension is case insensitive.
    dirname, ext = os.path.splitext(img_path)
    img_name = os.path.basename(img_path)
    if ext and not os.path.isfile(dirname + ext.lower()):
        img_path = dirname + ext.upper()
        
    # Default values.
    info = {'Width': zmax, 'Height': zmax, 'Orientation': 1, 'FrameRate': zmax, \
            'Duration': zmax, 'Make': 'N/A','Model': 'N/A','DateTime': None,\
            'GPSInfo': None}
    
    # Updating values if img_path existed.
    if not os.path.isfile(img_path):
        print(f"Error reading EXIF from {img_path}.")
    else:
        with Image.open(img_path) as img:
            info['Width'] = img.width
            info['Height'] = img.height
            
            # Orientation, if present
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            orientation_key = None
            make = 'N/A'
            model = 'N/A'
            
            if exif_data is not None:
                # Find numeric tags for Orientation, Make, Model
                for k, v in ExifTags.TAGS.items():
                    if v == 'Orientation':
                        orientation_key = k
                    elif v == 'Make':
                        make_key = k
                    elif v == 'Model':
                        model_key = k
                    elif v == 'DateTime':
                        datetime_key = k
                        
                if orientation_key in exif_data:
                    info['Orientation'] = exif_data[orientation_key]
                else:
                    info['Orientation'] = 1  # default

                if make_key in exif_data:
                    make = exif_data[make_key]
                if model_key in exif_data:
                    model = exif_data[model_key]
                info['Make'] = make
                info['Model'] = model

                # Date/time
                if datetime_key in exif_data:
                    dt_str = exif_data[datetime_key]
                    try:
                        dt_str = dt_str.replace(':', ' ', 2)  
                        dt_parts = dt_str.split()
                        date_part = dt_parts[0:3]                              # Y M D
                        time_part = dt_parts[3] if len(dt_parts) > 3 else "00:00:00"
                        hours, minutes, seconds = time_part.split(':')
                        dt_object = datetime(
                            int(date_part[0]),
                            int(date_part[1]),
                            int(date_part[2]),
                            int(hours),
                            int(minutes),
                            int(seconds)
                        )
                        info['DateTime'] = dt_object
                    except:
                        info['DateTime'] = None
                else:
                    info['DateTime'] = None

                # GPS Info
                gps_info_key = None
                for k, v in ExifTags.TAGS.items():
                    if v == 'GPSInfo':
                        gps_info_key = k
                        break

                if gps_info_key in exif_data:
                    gps_data = exif_data[gps_info_key]
                    gps_info = {}
                    for t in gps_data:
                        tag_name = ExifTags.GPSTAGS.get(t, t)
                        if isinstance(gps_data[t], bytes):
                            val = list(gps_data[t])
                            gps_info[tag_name] = val
                            if len(val) == 1: 
                                gps_info[tag_name] = val[0]
                        else:
                            gps_info[tag_name] = gps_data[t]
                            
                    # Converting GPS from dms to degree.
                    gps_lon = gps_info.get('GPSLongitude',[0,0,0])
                    gps_lat = gps_info.get('GPSLatitude',[0,0,0])
                    lon_deg = dms2degree(gps_lon[0], gps_lon[1], gps_lon[2])
                    lat_deg = dms2degree(gps_lat[0], gps_lat[1], gps_lat[2])   
                    lat_ref = gps_info.get('GPSLatitudeRef',0)
                    lon_ref = gps_info.get('GPSLongitudeRef',0)
                    if lon_ref == 'W': lon_deg = -lon_deg
                    if lat_ref == 'S': lat_deg = -lat_deg
                    gps_info['GPSLongitude'] = lon_deg
                    gps_info['GPSLatitude'] = lat_deg
                    
                    # Checking gps data format.
                    keys = ['GPSLatitude', 'GPSLongitude', 
                            'GPSAltitudeRef', 'GPSAltitude']
                    for key in keys:
                        v = gps_info.get(key, 0)
                        if isinstance(v,(float,int)):
                            if math.isnan(v): v = 0
                        else:
                            v = 0
                        gps_info[key] = int(v) if key == 'GPSAltitudeRef' else v
                        
                    info['GPSInfo'] = gps_info
                else:
                    info['GPSInfo'] = None
            else:
                # No EXIF
                if 'dji' in img_name.lower():
                    make = 'DJI'
                    model = 'FC3411'
                info['Orientation'] = 1
                info['Make'] = make
                info['Model'] = model
                info['DateTime'] = None
                info['GPSInfo'] = None
    return info

#%% Extracting the EXIF information of a video.
def readEXIFVideo(video_path: str, zmax=-9999) -> dict:
    """
    Extract “EXIF”-style metadata from a video file using ExifToolHelper.
    Returns a dict with keys:
      Width, Height, Orientation, Make, Model, DateTime, GPSInfo
    """
    
    dirname, ext = os.path.splitext(video_path)
    if ext and not os.path.isfile(dirname + ext.lower()):
        video_path = dirname + ext.upper()
        
    # Define path to exteral tool: exiftool.exe.
    system = platform.system().lower()
    if system == 'windows':
        exif_name = "exiftool-13.40_32"
        exif_os = 'exiftool_windows' 
        exif_folder = os.path.join(os.path.dirname(mathm_path), "exes", exif_os, exif_name)
        exif_zip = exif_folder + ".zip"
        exif_path = os.path.join(exif_folder, exif_name, r"exiftool(-k).exe")
        if not os.path.isfile(exif_path):
            with zipfile.ZipFile(exif_zip, 'r') as zip_ref:
                zip_ref.extractall(exif_folder) 
                
    else:
        exif_name = "Image-ExifTool-13.34"
        exif_os = 'exiftool_linux' 
        exif_folder = os.path.join(os.path.dirname(mathm_path), "exes", exif_os, exif_name)
        exif_path = os.path.join(exif_folder, "exiftool")
        
        # If exiftool is missing, extract it
        if not os.path.isfile(exif_path):
            exif_tar = exif_folder + ".tar.gz"
            if not os.path.isfile(exif_tar):
                raise FileNotFoundError(f"ExifTool archive not found: {exif_tar}")
    
            # Make sure the folder exists
            os.makedirs(exif_folder, exist_ok=True)
    
            # Extract into parent (so it creates exif_folder)
            with tarfile.open(exif_tar, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(exif_folder))
    
            # Make sure the binary is executable
            os.chmod(exif_path, os.stat(exif_path).st_mode | 0o111)

    info = {'Width': zmax, 'Height': zmax, 'Orientation': 1, 'FrameRate': zmax, \
            'Duration': zmax, 'Make': 'N/A','Model': 'N/A','DateTime': None,\
            'GPSInfo': None}
        
    if os.path.isfile(video_path) and os.path.isfile(exif_path):
        with ExifToolHelper(executable=exif_path) as et:
            #md = et.execute_json("-ee", "-n", video_path)
            md = et.get_metadata(video_path)[0]
            
        # Dimensions.
        w = md.get('QuickTime:ImageWidth')
        h = md.get('QuickTime:ImageHeight')
        rot =  md.get('Composite:Rotation')
        if rot == 90 or rot == -90: 
            w = md.get('QuickTime:ImageHeight'); h = md.get('QuickTime:ImageWidth')
        info['Width'] = w
        info['Height'] = h
        
        # Orientation.
        info['Orientation'] = 1
         
        # Frame rate and total duration.
        info['FrameRate'] = md.get('QuickTime:VideoFrameRate')
        info['Duration'] = md.get('QuickTime:MediaDuration')
        
        # Make, model, time, time zone shift.
        info['Make']  = md.get('QuickTime:Make')
        info['Model'] = md.get('QuickTime:Model')
        if 'dji' in video_path.lower():
            info['Make'] = 'DJI'
            info['Model'] = 'FC3411'
            dt_str = md.get('QuickTime:CreateDate')
            dt_mod = md.get('File:FileModifyDate')
            dt_off = int(dt_mod.split('-')[1].split(':')[0])
        else:
            dt_str = md.get('QuickTime:CreationDate')
            dt_str = dt_str[:-6]
            dt_off = 0
            
        # Converting datestr to obj.
        if 'T' in dt_str:
            dt_obj = datetime.fromisoformat(dt_str.rstrip('Z'))
        else:
            dt_obj = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
        dt_obj = dt_obj - timedelta(hours=dt_off)                              # Ajusting time to local time.
        info['DateTime'] = dt_obj    
        
        # GPSInfo
        gps = {}
        for tag in ('GPSLatitude','GPSLongitude','GPSAltitude',\
                    'GPSLatitudeRef','GPSLongitudeRef','GPSAltitudeRef'):
            if 'Composite:' + tag in md: gps[tag] = md['Composite:' + tag]  
        info['GPSInfo'] = gps or None
    else:
        print(f"Error reading metadata from {video_path}")
        
    return info

#%% Extracting the EXIF information of a photo or video.
def readEXIF(filename: str, zmax=-9999) -> dict:
    photo_ext = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"}
    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}
    #filename = Path(filename)
    ext = Path(filename).suffix.lower()
    
    if ext in photo_ext:
        info = readEXIFPhoto(filename)
    elif ext in video_ext:
        info = readEXIFVideo(filename)
    else:
        info = {'Width': zmax, 'Height': zmax, 'Orientation': 1, 'FrameRate': zmax,
                'Duration': zmax, 'Make': 'N/A','Model': 'N/A','DateTime': None,
                'GPSInfo': None}
    return info

#%% Reading photo EXIF information.
def getExif(pv_name):
    exif={}
    image_file = Image.open(pv_name)
    for k, v in image_file.getexif().items():tag=TAGS.get(k); exif[tag]=v
    return exif

#%% Loading a single drone flight record data and correcting the data.        #
# DJI AIR 2S thickness: 0.065 m (from the bottom to the top).                 #    
# height_above_takeoff(meter) and ascent(meter) are identical. Their value    #
# is 1.1 m during the auto take off stage. This value is the same as the      #
# one displayed on cellphone app. This value is also closest to video log     #
# file vlaue; height_sonar(meter) is 1.2 m during the auto takeoff stage,     #
# consistent with DJI APP descriptions (auto takeoff 1.2 m). It is also       #
# consistent with tap measurement; height_above_ground_at_drone_location:     # 
# it is lowest. It varies between 1.1 - 1 m during auto take off.It equals    #
# altitude_above_seaLevel minus ground_elevation_at_drone_location; zSpeed    #
# integrated height: its value is 1.05 m during the take off. Note that       #
# the velocity sensor may be 3-4 cm above the ground. This means the real     #
# height will be around 1.1 m. This is consistant the height above takeoff    #
# or ascent values. Overall, the sonar height is the most accurate, which     # 
# is close to DJI App take off target 1.2 m and measured values. However,     #
# the height above_takeoff is what displayed in the cellphone.                #
# FlyState code meaning:                                                      #
# 41: Motors_Started; 11: AutoTakeoff; 6: P-GPS: 31: Sport; 38: Tripod;       #
# 15: Go_Home; 12: AutoLanding; 33: Confirm_Landing.                          #
def readFCSingle(filename, writedata=True):
    # parse path, name, extension
    fpath, fname_ext = os.path.split(filename)
    fname, ext = os.path.splitext(fname_ext)

    # read CSV preserving column names
    fc0 = pd.read_csv(filename)
    fc0.columns = (
        fc0.columns
           .str.strip()  # remove leading/trailing spaces
           .str.replace(r'\(mph\)', '(mph)', regex=True)
           .str.replace(r'\(m/s\)', '(mps)', regex=True)
    )
    fnames = fc0.columns.tolist()

    # numeric block cols 3:50 → indices 2..49
    fc0_num = fc0.iloc[:, 2:50].to_numpy()
    mask_zero = (fc0_num[:, :2] == 0).any(axis=1)
    fc0_num[mask_zero, :2] = np.nan

    #nf = len(fnames)

    # ———————————————————————————
    # 2. Recompute subseconds & per‐second averages
    # ———————————————————————————
    # unique timestamps and mapping
    t0 = pd.to_datetime(fc0[fnames[1]])
    uniq_t0, ic = np.unique(t0.values, return_inverse=True)

    # run‐length encode ic
    st, na, val = numCount(ic)
    na_unq = np.unique(na)
    tt0 = np.arange(len(st))

    tc0 = np.zeros(len(fc0))
    fc0_mean = np.full((len(na), fc0_num.shape[1]), np.nan)

    for run_len in na_unq:
        runs = np.where(na == run_len)[0]
        if run_len <= 0 or runs.size == 0:
            continue
        # start‐indices for these runs
        starts = st[runs]
        # integer‐second part
        ti = tt0[runs][:, None]
        # fractional offsets
        frac = np.linspace(0, 1, run_len + 1)[:-1]
        dti = np.tile(frac, (runs.size, 1))
        # assign recomputed time
        rows = starts[:, None] + np.arange(run_len)
        tc0[rows.flatten()] = (ti + dti).flatten()
        # per‐second averages
        for j in range(fc0_num.shape[1]):
            block = fc0_num[rows, j]  # shape (runs.size, run_len)
            fc0_mean[runs, j] = block.mean(axis=1)

    # overwrite first column with recomputed time
    fc0[fnames[0]] = tc0
    #fc0 = fc0.iloc[st[1]:st[-1]+1].reset_index(drop=True)
    fc0 = fc0.iloc[st[1]:st[-1]+1]
    fc0[fnames[0]] -= fc0[fnames[0]].iloc[0]
    
    # ———————————————————————————
    # 3. Build second (averaged) table fc1
    # ———————————————————————————
    cols1 = fnames[:-2]  # all except last two original columns
    #fc1 = pd.DataFrame(index=range(len(na)), columns=cols1)
    fc1 = pd.DataFrame(0,index=range(len(na)),columns=cols1)
    fc1[fnames[0]] = tc0[st]
    fc1[fnames[1]] = uniq_t0

    for j in range(fc0_num.shape[1]):
        col = fnames[j + 2]
        mapping = dict(zip(fc0[fnames[0]], fc0[col]))
        if col in ('isPhoto', 'isVideo'):
            # copy flags for integer‐second rows
            mask = fc1[fnames[0]].isin(mapping)
            fc1.loc[mask, col] = fc1.loc[mask, fnames[0]].map(mapping)
        else:
            fc1[col] = fc0_mean[:, j]

    # drop the first row and re‐zero
    fc1 = fc1.iloc[1:].reset_index(drop=True)
    fc1[fnames[0]] -= fc1[fnames[0]].iloc[0]

    # package for further processing
    fcs = [fc0.iloc[:, :50].reset_index(drop=True),
           fc1.reset_index(drop=True)]

    # ———————————————————————————
    # 4. Determine unit system & compute velocity‐integration height
    # ———————————————————————————
    unit_sys = 1
    if re.search('feet', fnames[4], flags=re.IGNORECASE): unit_sys = 0
    
    # if unit_sys == 0:
    #     # imperial → metric conversions
    #     keynames = ['flycStateRaw', 'xSpeed(mph)', 'ySpeed(mph)', 'zSpeed(mph)']
        
    #     # boolean mask: in‐flight (state==6) & zero ground‐speed
    #     # tf = (
    #     #     (fc0[keynames[0]] == 6) &
    #     #     (fc0[keynames[1]] == 0) &
    #     #     (fc0[keynames[2]] == 0) &
    #     #     (fc0[keynames[3]] == 0)
    #     # )
    #     # time vector [s]
    #     t = fc0.iloc[:, 0].to_numpy()
    #     # vertical speed [m/s] (mph → m/s)
    #     uz = fc0.iloc[:, 21].to_numpy() * 0.44704
    #     # height by cumulative trapezoidal integration
    #     hvt = cumtrapz(-uz, t)
    #     # barometric/sonar altitude [m] (ft → m)
    #     #hso = fc0.iloc[:, 8].to_numpy() * 0.3048
    
    # else:
    #     # metric (no conversion)
    #     keynames = ['flycStateRaw', 'xSpeed(mps)', 'ySpeed(mps)', 'zSpeed(mps)']
    #     # tf = ((fc0[keynames[0]] == 6) &
    #     #     (fc0[keynames[1]] == 0) &
    #     #     (fc0[keynames[2]] == 0) &
    #     #     (fc0[keynames[3]] == 0))
    #     t = fc0.iloc[:, 0].to_numpy()
    #     uz = fc0.iloc[:, 21].to_numpy()         # already m/s
    #     hvt = cumtrapz(-uz, t)
    #     #hso = fc0.iloc[:, 8].to_numpy()         # already m
    
    zz0 = 0.0

    # ———————————————————————————
    # 5. Final per‐mode processing (V and P)
    # ———————————————————————————
    FCKeyWords = ['V', 'P']
    FCT = [None] * 2
    fct = [None] * 2

    for iv in range(2):
        # select same indices as MATLAB: [1,3:12,48,20:22,49,50,26,27]
        cid = ([0] +
               list(range(2, 12)) +
               [47] +
               list(range(19, 22)) +
               [48, 49, 25, 26])
        df = fcs[iv]
        sel_names = df.columns[cid]
        arr = df.iloc[:, cid].to_numpy()

        # unit conversions
        if unit_sys == 0:
            units = np.full(arr.shape[1], 0.3048)
            units[[0, 1, 2, 16, 17, 18]] = 1.0
            units[[8, 12, 13, 14]] = 0.44704
        else:
            units = np.ones(arr.shape[1])
        arr = arr * units

        # rebuild timetable with datevec
        t0 = pd.to_datetime(df[fnames[1]]).dt.tz_localize('UTC') \
                                     .dt.tz_convert('America/Vancouver')
        dv = np.column_stack([
            t0.dt.year.values,
            t0.dt.month.values,
            t0.dt.day.values,
            t0.dt.hour.values,
            t0.dt.minute.values,
            t0.dt.second.values
        ])

        # assemble final numeric array
        fc_comb = np.column_stack((arr[:,0],dv,arr[:,1:]))
        h_vt = cumtrapz(-fc_comb[:, 20], fc_comb[:, 0]) + zz0
        fc_final = np.column_stack((fc_comb, h_vt.reshape(-1,1),np.ones((fc_comb.shape[0],1))*zz0))

        # build new column names
        namesnew = (['time(second)', 'Year', 'Month', 'Day',
                     'Hour', 'Minute', 'Second'] +
                    list(sel_names[1:]) +
                    ['height_velocity_integration(meter)',
                     'height_correction(meter)'])
        namesnew = [s.replace('feet', 'meter')
                        .replace('mph', 'mps')
                    for s in namesnew]

        # create DataFrame
        FC_df = pd.DataFrame(fc_final, columns=namesnew)

        # write CSV if requested
        tokens = fname.split('-')
        prefix = '_'.join(tokens[:-2])
        outname = f"{prefix}_FC{FCKeyWords[iv]}{ext}"
        outpath = os.path.join(fpath, outname)
        if os.path.isfile(outpath): os.remove(outpath)
        if writedata: FC_df.to_csv(outpath, index=False)

        FCT[iv] = FC_df
        fct[iv] = fc_final.astype(np.float64)

    return fct, FCT, namesnew

#%% Correcting multiple drone flight record data.                             #
def readFC(filename, writedata=True):
    FCKeyWords = ['V', 'P']
    keyword = 'Flight-Airdata.csv'
    fct = []; FCT = []; names = []
    
    # Single‐file case: just pass through to readFC
    if os.path.isfile(filename): 
        fct, FCT, names = readFCSingle(filename, writedata)
        
    # Folder case
    if os.path.isdir(filename):
        # find matching files
        all_files = os.listdir(filename)
        fcfiles = [f for f in all_files if re.search(keyword, f, re.IGNORECASE)]
        fcfiles = natsorted(fcfiles)
        
        if len(fcfiles)>0:
            # read first file (no write)
            fc1, _, names = readFCSingle(os.path.join(filename, fcfiles[0]), 'no')
            fcv, fcp = fc1[0], fc1[1]
    
            # capture the reference date‐vector for each stream
            t0_v = fcv[0, 1:7].astype(int)
            t0_p = fcp[0, 1:7].astype(int)
    
            # loop over remaining files
            for fname in fcfiles[1:]:
                fci, _, names = readFCSingle(os.path.join(filename, fname), False)
                fcvi, fcpi = fci[0], fci[1]
    
                # compute elapsed‐time in seconds between date‐vectors
                dt_v = (
                    datetime(*fcvi[0, 1:7].astype(int))
                    - datetime(*t0_v)
                )
                stv = dt_v.total_seconds()
    
                dt_p = (
                    datetime(*fcpi[0, 1:7].astype(int))
                    - datetime(*t0_p)
                )
                stp = dt_p.total_seconds()
    
                # offset the first column (time)
                fcvi[:, 0] += stv
                fcpi[:, 0] += stp
    
                # concatenate
                fcv = np.vstack((fcv, fcvi))
                fcp = np.vstack((fcp, fcpi))
    
            fct = [fcv, fcp]
            FCT = []
    
            # for each channel, wrap in a DataFrame and optionally write out
            for ic, arr in enumerate(fct):
                df = pd.DataFrame(arr, columns=names)
                
                # build output name from first file’s name
                parts = fcfiles[0].split('-')
                basename = '_'.join(parts[:-2])
                outname = f"{basename}_FC{FCKeyWords[ic]}.csv"
                outpath = os.path.join(filename, outname)
                if os.path.exists(outpath): os.remove(outpath)
                if writedata: df.to_csv(outpath, index=False)
    
                FCT.append(df)

    return fct, FCT, names

#%% Read and merge CSVs inside a single folder.
def readFile(workdir, filtername='', overwrite=False, writedata=False, by='', 
             excludes='', pattern='_*.csv', unique=False, dropna=True, 
             ascending=True, keyword='Scales', version='V3'):
    
    prefix  = f"{keyword}_{version}"
    cfolder = os.path.basename(workdir.rstrip(os.sep))
    outname = f"{prefix}_{cfolder}.csv"
    outpath = os.path.join(workdir, outname)
    
    df = pd.DataFrame()
    if (not os.path.isfile(outpath)) or overwrite:
        patt = f"{prefix}_{cfolder}{pattern}"
        fps = natsorted([fp for fp in glob.glob(os.path.join(workdir, patt))])
        if len(fps) == 0:
            print(f"{cfolder}: no files matching {patt}, ...")
            
        for fp in fps:
            if os.path.getsize(fp)>10:
                dfi = pd.read_csv(fp,keep_default_na=False)
                df = pd.concat([df, dfi], ignore_index=True)
                del dfi
            else:
                print(f"{cfolder}: {os.path.basename(fp)} is empty, ...")

            
        # Do this only for non-empty df.
        if not df.empty:
            #df = df.drop_duplicates()                                          # Remove duplicated rows.
            if filtername and (filtername in df.columns):
                df[filtername] = pd.to_numeric(df[filtername], errors="coerce")
                df = df[df[filtername] > 0]                                    # Keep only data whose value larger tha 0 for filtername.
                
            # sort
            nkey = natsort_keygen()
            if isinstance(by, str): by = [by]
            if by != ['']:
                df = df.sort_values(by= by,ascending= [ascending] * len(by),
                    key=lambda col: col.map(nkey) if col.name == "Name" else col
                ).reset_index(drop=True)
            
            # Remove rows with repeated name if provied.
            if unique: df = df.loc[df.groupby("Name")["L1_marker_meter"].\
                                   idxmax()].reset_index(drop=True)
            if dropna: df = df.dropna(how='all')                               # Remove all nan data.
            if not df.empty and writedata:
                df.to_csv(outpath,index=False)                                 # Write data to disk.
    else:
        if os.path.getsize(outpath)>10:
            df = pd.read_csv(outpath,keep_default_na=False)
            
    return df

#%% Recursively call readFile on all subfolders and merge into one DataFrame. #
def readFiles(workdir, filtername='', overwrite=False, writedata=False, by='', 
             excludes='', pattern='_*.csv', unique=True, dropna=True, 
             ascending=True, keyword='Scales', version='V3'):
    
    default_excludes = {".", ".."}
    prefix  = f"{keyword}_{version}"
    patt = f"{prefix}_subfoldername{pattern}"
    cfolder = os.path.basename(workdir.rstrip(os.sep))
    outname = f"{prefix}_{cfolder}.csv"
    outpath = os.path.join(workdir, outname)

    # normalize excludes (folder names)
    if isinstance(excludes, str):
        excludes = {excludes.lower()} if excludes else set()
    else:
        excludes = {str(e).lower() for e in (excludes or [])}
    all_excludes = {d.lower() for d in default_excludes} | excludes

    df = pd.DataFrame()
    if (not os.path.isfile(outpath)) or overwrite:
        folders = getFolders(workdir, excludes=all_excludes)                   # Get folders in this folder.
        for folder in folders:
            subdir = os.path.join(workdir, folder)
            dfi = readFile(
                workdir=subdir,
                filtername=filtername,
                overwrite=True,
                writedata=False,                                               # Do not write data to local disk.
                by=by,
                excludes='',
                pattern=pattern,
                unique=unique,
                dropna=dropna,
                ascending=ascending,
                keyword=keyword,
                version=version)
            if not dfi.empty: 
                df = pd.concat([df, dfi], ignore_index=True)
            del dfi
        df = df.drop_duplicates()                                              # Remove duplicated rows.
        
        if df.empty:
            print(f'{cfolder}: no files matching {patt}')
        else:
            df.to_csv(outpath, index=False)
    else:
        if os.path.getsize(outpath)>10: 
            df = pd.read_csv(outpath,keep_default_na=False)
    print()
    
    return df

#%%