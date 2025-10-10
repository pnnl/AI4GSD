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
import os, pathlib
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline, \
    Akima1DInterpolator
    
#%% Calculating the start, number, and value of unique values in an array.    #
# A could be an array with N*0 and N*1 dimension.                             #
# Performance: 0.04 s for 1M data, 0.22 s for 10M data, too slow for 100M data#
def numCount(A, tol=0):
    Idx = np.arange(A.shape[0])
    df = np.abs(np.diff(A, axis=0)) > tol
    df = np.insert(df, 0, True)
    df = np.insert(df, len(df), True)
    CheckI = np.where(df)[0]
    st = Idx[CheckI[:-1]]
    na = np.diff(CheckI)
    val = A[st]
    return st, na, val

#%% This function calcualtes the empirical cumulative distribution function   #
# This function generates results identical to the ecdf in Matlab and         #
# ecdfm2 function under Basics folder. y is requried as N*1 or N*M            #
# The by area algorithm is P(X) = sum(a(x<X))/sum(a) with a and x denoting    #
# the area and size of each value. X is the unique values in x.               #
def ecdfm(y,option='bycount',tol=0):
    f = []; x = []; area_sum = [];
    if option.lower() == 'bycount':
        if y.ndim == 1:
            st, na, val = numCount(np.sort(y),tol)
            f = np.concatenate(([0], np.cumsum(na) / y.shape[0]))
        else:
            y = np.sort(y, axis=0)
            st, na, val = numCount(y[:, 0],tol)
            f = np.concatenate(([0], np.cumsum(na) / y.shape[0]))
        x = np.concatenate((val[0:1 :], val))
    elif option.lower() == 'byarea':
        if y.ndim == 1:
            raise 'Area data not found for by area cumulative distribution .'
        else:
            y = np.sort(y, axis=0)
            st, na, val = numCount(y[:, 0],tol)
            area_sum = np.zeros(len(st), dtype=float)
            single_occ_mask = (na == 1)
            area_sum[single_occ_mask] = y[st[single_occ_mask], 1]
            
            # Loop from k=2 to max(na)
            for k in range(2, int(np.max(na)) + 1):
                tf = np.where(na == k)[0]  # indices where na == k
                nr = len(tf)
                if nr == 0:
                    continue
                
                row_offsets = np.arange(1, k + 1)  # [1, 2, ..., k]
                tiled_offsets = np.tile(row_offsets, (nr, 1))  # shape (nr, k)
                base_indices = (st[tf] - 1).reshape(-1, 1)
                idr = (base_indices + tiled_offsets).T.flatten().astype(int)
                chunk = y[idr, 1].reshape(k, -1)
                area_sum[tf] = np.sum(chunk, axis=0)
            f = np.concatenate(([0], np.cumsum(area_sum) / np.sum(y[:, 1])))
            x = np.concatenate((val[0:1 :], val))
    else:
        raise 'Wrong weight option.'
    return f, x, area_sum
    
#%% Listing all files and return extension and isvideo indicator.             #
def getFilesStable(WorkDir,vext = ['.mp4','.mov','.avi'],\
             pext = ['.jpg','.jpeg','.png','.tiff','.tif']):
    allfiles = pathlib.Path(WorkDir)
    files = []; stems = []; ext = []; ispv = []
    for item in allfiles.iterdir():
        if item.is_file() and item.name[0] != '.':
            files.append(item.name)
            stems.append(item.stem)
            ext.append(item.suffix)
            ispv_f = -1
            if item.suffix.lower() in pext: ispv_f = 0
            if item.suffix.lower() in vext: ispv_f = 1
            ispv.append(ispv_f)
            
    return files,stems,ext,ispv

#%% New version of get files with natsorted.
def getFiles(WorkDir,vext = ['.mp4','.mov','.avi'],\
             pext = ['.jpg','.jpeg','.png','.tiff','.tif'],sortname=True):
    p = pathlib.Path(WorkDir)
    items = [f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]
    if sortname: items = natsorted(items, key=lambda x: x.name)
    
    files = [f.name for f in items]
    stems = [f.stem for f in items]
    ext   = [f.suffix for f in items]
    ispv  = [
        0 if f.suffix.lower() in pext else
        1 if f.suffix.lower() in vext else
       -1
        for f in items
    ]
    return files, stems, ext, ispv

#%% Obtaining folder names under a given path with excluding option.
def getFolders(path, excludes='', includepattern='', excludepattern=''):
    default_excludes = {".", ".."}
    
    # Normalize excludes to a set of lowercase strings
    if isinstance(excludes, str):
        user_excludes = {excludes.lower()} if excludes else set()
    else:
        user_excludes = {u.lower() for u in excludes}
    all_excludes = {d.lower() for d in default_excludes} | user_excludes

    folders = [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
        and name.lower() not in all_excludes
    ]

    # Case-insensitive substring filtering
    if includepattern:
        folders = [f for f in folders if includepattern.lower() in f.lower()]
    if excludepattern:
        folders = [f for f in folders if excludepattern.lower() not in f.lower()]

    return natsorted(folders)

#%% Converting box center to corners.
def center2corners(center_xy, wh):
    x1 = center_xy[:, 0] - wh[:, 0] / 2.0
    y1 = center_xy[:, 1] - wh[:, 1] / 2.0
    x2 = center_xy[:, 0] + wh[:, 0] / 2.0
    y2 = center_xy[:, 1] + wh[:, 1] / 2.0

    return np.stack([x1, y1, x2, y2], axis=1)

#%% Redefining linspace to be consistant with Matlab.
def linspace(start,stop, npoints):
    values = np.zeros(npoints,dtype=float)
    if npoints == 1:
        values[0] = stop
    else:
        values[:] = np.linspace(start, stop, npoints)
    return values

#%% Checking if elements in A are in B.
def ismembertol(A, B, tol=None, byrows=False):
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Determine default tolerance if not provided.
    if tol is None:                                                
        if A.dtype == np.float32 or B.dtype == np.float32:
            tol = 1e-6
        else:
            tol = 1e-12
            
    # Compute the scale factor: maximum absolute value over both A and B.
    max_val = np.max(np.abs(np.concatenate((A.ravel(), B.ravel()))))
    scaled_tol = tol * max_val

    if not byrows:
        # Element-wise comparison.
        A_flat = A.ravel()
        B_flat = B.ravel()
        LIA = np.zeros(A_flat.shape, dtype=bool)
        LocB = np.zeros(A_flat.shape, dtype=int)                               # 0 indicates no match, MATLAB-style
        
        # Find indices in B where the absolute difference is within scaled_tol.
        for i, a_val in enumerate(A_flat):
            diffs = np.abs(a_val - B_flat)
            idx = np.where(diffs <= scaled_tol)[0]
            if idx.size > 0:
                LIA[i] = True
                LocB[i] = idx[0] + 1                                           # +1 for MATLAB-style indexing
            else:
                LIA[i] = False
                LocB[i] = 0
                
        # Reshape the outputs to the shape of A.
        LIA = LIA.reshape(A.shape)
        LocB = LocB.reshape(A.shape)
        
    else:
        # Row-wise comparison.
        # Ensure both A and B are 2D.
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)
        n_rows_A = A.shape[0]
        LIA = np.zeros(n_rows_A, dtype=bool)
        LocB = np.zeros(n_rows_A, dtype=int)  # 0 indicates no match
        
        for i in range(n_rows_A):
            # For each row in A, check every row in B.
            for j in range(B.shape[0]):
                # Check if all elements of the rows differ by no more than scaled_tol.
                if np.all(np.abs(A[i, :] - B[j, :]) <= scaled_tol):
                    LIA[i] = True
                    LocB[i] = j + 1                                            # MATLAB-style index
                    break  # Only the first match is recorded.
                    
    LocB = LocB - 1
    return LIA, LocB

#%%
def cumtrapz(y, x):
    """
    Cumulative trapezoidal integration of y with respect to x.
    Returns an array same length as y, starting with 0.
    """
    dx = np.diff(x)
    avg = (y[:-1] + y[1:]) / 2
    return np.concatenate(([0], np.cumsum(avg * dx)))

#%% Equivalent to the interp1 in Matlab.                                      #
# x: (n,:) array like; y: (n,) or (n,m) array like; method: linear, nearest,  #
# previous, next, zero, slinear, quadratic, cubic, pchip, makima, cspline.    #
def interp1(x, y, xi, method='linear', extrapolate=False, 
            fill_value=np.nan, assume_sorted=False):
    
    # Converting input x, y, xi as array type.
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)
    
    # Checking if size of x and y are consistent.
    if x.size != y.shape[0]:
         raise ValueError("len(x) must equal y.shape[0].")
    
    # Sort x and y if assume_sorted is False.
    if not assume_sorted:
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx, ...]
        
    # 2D columns along axis=0 (n, m)
    y2d = y if y.ndim > 1 else y[:, None]
    
    # Methods handled by interp1d directly
    methods = {'linear', 'nearest', 'previous', 'next', \
               'zero','slinear', 'quadratic', 'cubic'}
        
    method = method.lower()
    if method in methods:
        f = interp1d(x, y2d, kind=method, axis=0, bounds_error=False, 
                     fill_value=('extrapolate' if extrapolate else fill_value), 
                     assume_sorted=True)
        yi = f(xi)
    elif method == 'pchip':
        f = PchipInterpolator(x, y2d, axis=0, extrapolate=extrapolate)
        yi = f(xi)
        if not extrapolate:
            oob = (xi < x[0]) | (xi > x[-1])
            if np.any(oob):
                if np.ndim(fill_value) == 0:
                    yi[oob, ...] = fill_value
                else:
                    yi[oob, ...] = np.nan if np.ndim(fill_value) else fill_value
                    
    elif method == 'makima':
        f = Akima1DInterpolator(x, y2d, axis=0)
        yi = f(xi)
        if not extrapolate:
            oob = (xi < x[0]) | (xi > x[-1])
            if np.any(oob):
                yi[oob, ...] = fill_value if np.ndim(fill_value) == 0 else np.nan

    elif method == 'cspline':
        f = CubicSpline(x, y2d, axis=0, extrapolate=extrapolate)
        yi = f(xi)
        if not extrapolate:
            oob = (xi < x[0]) | (xi > x[-1])
            if np.any(oob):
                yi[oob, ...] = fill_value if np.ndim(fill_value) == 0 else np.nan

    else:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Choose from {sorted(methods | {'pchip','makima','cspline'})}."
        )

    if y.ndim == 1: yi = yi[..., 0]
    
    return yi

#%% Convert dates into array.
def datevec(dates):
    # Converting strings, lists, arrays, datetime to pandas datatime format.
    dt = pd.to_datetime(dates)   
                             
    # Converting dt to pandas DateTimeIndex if not converted.
    if not isinstance(dt, pd.DatetimeIndex):
        dt = pd.DatetimeIndex([dt])
        
    # Convert dt to array format.
    dt =  np.column_stack([
        dt.year,dt.month,dt.day,dt.hour,dt.minute,
        dt.second + dt.microsecond/1.0e6])

    return dt

#%% Compute the seconds between t1 and t2 (t2-t1).                            #
def etime(t1, t2):
    t1 = np.atleast_2d(t1)
    t2 = np.atleast_2d(t2)
    
    if t1.shape[0] == 1 and t2.shape[0] > 1: t1 = np.tile(t1, (t2.shape[0],1))
    if t1.shape[0] > 1 and t2.shape[0] == 1: t2 = np.tile(t2, (t1.shape[0],1))
    if t1.shape[0] != t2.shape[0]:
        raise ValueError("Row count of t1 must match row count of t2.")
        
    dt1 = pd.to_datetime({
        'year':   t1[:,0].astype(int),
        'month':  t1[:,1].astype(int),
        'day':    t1[:,2].astype(int),
        'hour':   t1[:,3].astype(int),
        'minute': t1[:,4].astype(int),
        'second': t1[:,5]
    })
    dt2 = pd.to_datetime({
        'year':   t2[:,0].astype(int),
        'month':  t2[:,1].astype(int),
        'day':    t2[:,2].astype(int),
        'hour':   t2[:,3].astype(int),
        'minute': t2[:,4].astype(int),
        'second': t2[:,5]
    })
    dt = (dt2 - dt1).dt.total_seconds().to_numpy()
    
    return dt

#%% Converting GPS from dms to degree.
def dms2degree(d, m, s):
    """
    Convert degrees-minutes-seconds to decimal degrees.
    d, m, s can be floats or arrays.
    """
    return d + m / 60.0 + s / 3600.0

#%% Checking if a file exists with extension case in-sensitive.
def isfile(filename,mode=1):
    isfile = os.path.isfile(filename)
    if not isfile:
        basename, ext = os.path.splitext(filename)
        filename = basename + ext.swapcase()
        isfile = os.path.isfile(filename)
        
    return isfile if mode == 1 else (isfile, filename)

#%% Compute statistics between input x and y.                                 #
def stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    x = x[mask]
    y = y[mask]
    r = np.corrcoef(x, y)[0, 1]
    r2 = r**2
    nse = 1 - np.sum((y - x) ** 2) / np.sum((x - np.mean(x)) ** 2)
    ee = y - x
    me = np.mean(ee)
    mae = np.mean(abs(ee))
    rmse = np.sqrt(ee **2)
    pe = ee / x * 100
    mpe = np.mean(pe)
    mape = np.mean(np.abs(pe))
    rmse = np.sqrt(np.mean((y - x) ** 2))
    rmspe = np.sqrt(np.mean(pe**2))                                            # RMS of percent errors
    nrmse = rmse / np.mean(x) * 100
    
    return {"r2": r2, "nse": nse, 'me': me, 'mae': mae, 'rmse': rmse, \
            "mpe": mpe, "mape": mape, 'rmse': rmse,'nrmse':nrmse,'rmspe':rmspe}
        
#%% Code testing.
if __name__ == "__main__":
    #import os
    y1 = np.array([
        [0.2, 0.03],
        [0.1, 0.01],
        [0.1, 0.01],
        [0.3, 0.05],
        [0.4, 0.07]
    ])
    y2 = np.array([0.2,0.1,0.1,0.3,0.4])
    
    # # f1,x1,area1 = ecdfm(y1)
    # # f2,x2,area2 = ecdfm(y2)
    # # f4,x4,area4 = ecdfm(y1,'byarea')
    # # print(f1,x1,area1)
    # # print(f2,x2,area2)
    # # print(f4,x4,area4)
    
    # # Example usage for element-wise comparison:
    # A_elem = [1.0, 2.001, 3.0]
    # B_elem = [1.00001, 2.0, 4.0]
    # LIA_elem, LocB_elem = ismembertol(A_elem, B_elem, tol=1e-3, byrows=False)
    # print("Element-wise LIA:", LIA_elem)
    # print("Element-wise LocB:", LocB_elem)
    
    # # Example usage for row-wise comparison:
    # A_rows = np.array([[0, 0.268458, 0.067739, 0.01462, 0.028915],
    #                    [0, 0.628838, 0.743665, 0.072368, 0.11306]])
    # B_rows = np.array([[0, 0.268458, 0.067739, 0.01462, 0.028915],
    #                    [0, 0.48273, 0.5013, 0.11568, 0.248863]])
    # LIA_rows, LocB_rows = ismembertol(A_rows, B_rows, tol=1e-5, byrows=True)
    # print("Row-wise LIA:", LIA_rows)
    # print("Row-wise LocB:", LocB_rows)
    
    # # Encode the image with JPEG compression
    # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    
    # # Decode the compressed image
    # compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    # outpath = os.path.splitext(imgpath)[0] + '_compressed' + os.path.splitext(imgpath)[1]
    # cv2.imwrite(outpath,compressed_img,[cv2.IMWRITE_JPEG_QUALITY, quality])
    # psz = os.path.getsize(outpath)/1024**2
    # encoded_img.size/1024**2
    
    # #return decoded_img
    
    # # Example usage
    # original_img = cv2.imread(imgpath)
    # #compressed_img = compress_image('original_image.jpg', quality=50)
    
    # # Display results
    
    # cv2.imshow('Original', original_img)
    # cv2.imshow('Compressed', compressed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # filename = r'/home/chen200/F/NG/test/videos/IMG_9683.mov'
    # #filename = r'/home/chen200/F/NG/test/videos/IMG_9683'
    # print(isfile(filename))
    # WorkDir = r'/home/chen200/F/NG/20250627_LR_WalkingPhotoFieldTest/videos'
    # print(getFiles(WorkDir))
    # print(getFilesOld(WorkDir))
    























