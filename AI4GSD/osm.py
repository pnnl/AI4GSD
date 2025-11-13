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
import os, time

#%% Get folder and file names from a given path.
def dirlists(path):
    folders = []
    files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_dir():
                folders.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)
    folders.sort()
    files.sort()
    return folders, files

#%% Recursive identify files from level 1 to maximum level (depth).
def walklevel(base_path, folder_chain, depth,
              nLmax_c, localFolderEmpty, folders, files,
              nLmax, formatSpecs):
    """
    Recursive traversal that updates:
        - nLmax_c[0]  (max depth)
        - folders     (list of padded folder chains)
        - files       (list of [path, filename])
    Uses formatSpecs for printing, same as MATLAB logic.
    """

    # Update max depth as in MATLAB
    nLmax_c[0] = max(nLmax_c[0], depth)

    # Print the folder chain (Folder A>B>C...) exactly like MATLAB.
    fmt = formatSpecs.get(depth)
    if fmt is not None:
        print(fmt % tuple(folder_chain), end='')

    # Add this folder chain to `folders` (padded to nLmax with '')
    row = localFolderEmpty.copy()
    row[:depth] = folder_chain
    folders.append(row)

    # List the current directory: subfolders + files
    subfolders, local_files = dirlists(base_path)

    # Append files (same logic: [path, name])
    for fname in local_files:
        files.append([base_path, fname])

    # Recurse into each subfolder if depth < nLmax
    if depth < nLmax:
        for sub in subfolders:
            new_chain = folder_chain + [sub]
            new_path = os.path.join(base_path, sub)
            walklevel(new_path, new_chain, depth + 1,
                      nLmax_c, localFolderEmpty, folders, files,
                      nLmax, formatSpecs)


#%% List all folders and files to count total number of files.                #
def fileSummary(WorkDir,maxdepth=20):

    # Check directory exists.
    if not os.path.isdir(WorkDir):
        raise NotADirectoryError(f"WorkDir does not exist or is not a directory: {WorkDir}")
        
    # Print information.
    formatSpec0  = '=' * 60
    formatSpec1  = "\nAnalyzing folder structure.\n"
    formatSpec2  = "{}\n".format('=' * 60)
    formatSpec3  = "\n\nFolder %s (L1)"
    formatSpec4  = "\nFolder %s>%s"
    formatSpec5  = "\nFolder %s>%s>%s"
    formatSpec6  = "\nFolder %s>%s>%s>%s"
    formatSpec7  = "\nFolder %s>%s>%s>%s>%s"
    formatSpec8  = "\nFolder %s>%s>%s>%s>%s>%s"
    formatSpec9  = "\nFolder %s>%s>%s>%s>%s>%s>%s"
    formatSpec10 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec11 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec12 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec13 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec14 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec15 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec16 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec17 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec18 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec19 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec20 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec21 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec22 = "\nFolder %s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s>%s"
    formatSpec100 = "\n\nMax level of folder structure %s\n"
    formatSpec101 = "\nTotal file number %s\n"
    formatSpec102 = "\nTotal folder number %s\n"
    formatSpec103 = "\nTotal search time %s s\n"
    print(formatSpec0, end='')
    print(formatSpec1, end='')
    
    # Information for each depth.
    formatSpecs = {
        1: formatSpec3,
        2: formatSpec4,
        3: formatSpec5,
        4: formatSpec6,
        5: formatSpec7,
        6: formatSpec8,
        7: formatSpec9,
        8: formatSpec10,
        9: formatSpec11,
        10: formatSpec12,
        11: formatSpec13,
        12: formatSpec14,
        13: formatSpec15,
        14: formatSpec16,
        15: formatSpec17,
        16: formatSpec18,
        17: formatSpec19,
        18: formatSpec20,
        19: formatSpec21,
        20: formatSpec22,
    }
    
    start_time = time.time()
    localFolderEmpty = [''] * maxdepth
    folders = []                                                               # list of 20-element folder-name rows
    files = []                                                                 # list of [path, filename]
    nLmax_c = [0]                                                              # mutable depth tracker

    # Level 0: root directory files and first-level folders.
    foldersL0, filesL0 = dirlists(WorkDir)
    for fname in filesL0:
        files.append([WorkDir, fname])
        
    # For each level-1 folder, call the recursive function with depth=1
    for f0 in foldersL0:
        pathL1 = os.path.join(WorkDir, f0)
        walklevel(pathL1, [f0], depth=1,
                  nLmax_c=nLmax_c,
                  localFolderEmpty=localFolderEmpty,
                  folders=folders,
                  files=files,
                  nLmax=maxdepth,
                  formatSpecs=formatSpecs)
        
    ext = [os.path.splitext(f[1])[1] for f in files]                           # Sorting files by extension.
    files_ext_pairs = list(zip(files, ext))
    files_ext_pairs.sort(key=lambda fe: fe[1].lower())
    if files_ext_pairs:
        files, ext = zip(*files_ext_pairs)
        files = list(files)
        ext = list(ext)
    else:
        files, ext = [], []

    # Unique extensions and counts
    ext_lower = [e.lower() for e in ext]
    ext_unq = sorted(set(ext_lower))
    ftypes = []
    for e in ext_unq:
        count = sum(1 for x in ext_lower if x == e)
        ftypes.append([e, count])

    nf = [ft[1] for ft in ftypes]
    nft = sum(nf)
    
    # Optional: select .m, .csv, .png files as in MATLAB (kept for later use)
    mfiles = [f for (f, e) in zip(files, ext) if e.lower() == '.m']
    csvfiles = [f for (f, e) in zip(files, ext) if e.lower() == '.csv']
    pngfiles = [f for (f, e) in zip(files, ext) if e.lower() == '.png']
     
    # Print summary information.
    print(formatSpec100 % str(nLmax_c[0]), end='')
    print(formatSpec101 % str(nft),       end='')
    print(formatSpec102 % str(len(folders)), end='')
    
    print("\nFile types and counts:")
    for ext_name, count in ftypes:
        label = ext_name if ext_name else '(no ext)'
        print(f"{label:6s} : {count}")
        
    elapsed = time.time() - start_time
    print(formatSpec103 % f"{elapsed:.3f}", end='')
    print(formatSpec2, end='')

    return nft, files, ext, ftypes, mfiles, csvfiles, pngfiles

#%%
