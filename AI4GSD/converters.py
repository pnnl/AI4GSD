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
import numpy as np
import pandas as pd
import os, cv2, json, math, shutil, random
from pathlib import Path
from natsort import natsorted
from collections import defaultdict
from .mathm import getFiles, ecdfm

#%% Converting YOLO.prediction result to mask.
def yolo2mask(res, outFileName=''):
    H, W = res.orig_shape[:2]
    mask_img = np.zeros((H, W), dtype=np.uint8)
    if res.masks is not None: 
        masks = res.masks.data.cpu().numpy()  
        combined_mask = np.any(masks, axis=0)                                  # Convert data to bool.
        mask_img = (combined_mask.astype(np.uint8)) * 255                      # Convert data to unit8.
        
    if outFileName != '':
        save_dir = os.path.dirname(outFileName)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(outFileName,mask_img)
    return mask_img
 
#%% Converting YOLO.prediction result to txt.
def yolo2txt(res, outFileName=''):
    H, W = res.orig_shape[:2]
    lines = []
    if res.masks is not None: 
        lines = []
        cls_ids = res.boxes.cls.cpu().tolist()
        polygons = res.masks.xy
        for cid, poly in zip(cls_ids, polygons):
            poly_arr = np.array(poly, dtype=float)
            norm = poly_arr / [W, H]
            coords_str = " ".join(f"{v:.6f}" for xy in norm for v in xy)
            lines.append(f"{cid} {coords_str}")
            
    if outFileName != '':
        save_dir = os.path.dirname(outFileName)
        os.makedirs(save_dir, exist_ok=True)
        with open(outFileName, "w") as f:
            for line in lines:
                f.write(line + "\n")         
    return lines
                
#%% Converting YOLO.prediction result to json.
def yolo2json(res,outFileName=''):
    H, W = res.orig_shape[:2]
    shapes = []
    if res.masks is not None: 
        shapes = []
        cls_ids = res.boxes.cls.cpu().tolist()
        polygons = res.masks.xy
        class_names = [res.names[i] for i in sorted(res.names.keys())]
        for cid, poly in zip(cls_ids, polygons):
            pts = [[float(x), float(y)] for x, y in poly]
            shapes.append({
                "label":      class_names[int(cid)],
                "points":     pts,
                "group_id":   None,
                "shape_type": "polygon",
                "flags":      {}
            })
    
    js_data = {
        "version":     "5.4.1", 
        "flags":       {}, 
        "shapes":      shapes, 
        "imagePath":   os.path.basename(res.path),
        "imageData":   None,
        "imageHeight": H,
        "imageWidth":  W
        }
    
    if outFileName != '':
        save_dir = os.path.dirname(outFileName)
        os.makedirs(save_dir, exist_ok=True)
        with open(outFileName, "w", encoding="utf-8") as jf:
            json.dump(js_data, jf, ensure_ascii=False, indent=2)   
    return js_data

#%%
def labelCorrectionOld(filename,classid=0, overwrite=False,writefolder='labels_corrected'): 
    if os.path.isfile(filename):
        files = [os.path.basename(filename)]
        filename = os.path.dirname(filename)
    else:
        files = [f for f in os.listdir(filename) if f.endswith(".txt")]
        files = list(np.setdiff1d(files, ['classes.txt']))

    fmt = ["%d","%.6f","%.6f","%.6f","%.6f"]
    outfolder = filename + os.sep + writefolder
    if len(files)>0:
        os.makedirs(outfolder, exist_ok=True)
        for file in files:
            outname = outfolder + os.sep + file
            if not os.path.isfile(outname) and not overwrite:
                data = np.loadtxt(filename + os.sep + file, dtype=float, ndmin=2)
                data[:,0] = classid
                np.savetxt(outname, data, fmt=fmt)
    return

#%% Correcting labels in a folder to make it self-consistent.                 #
def labelCorrectionLevel1(foldername,labelsource='labels_org',excludes=None):
    formatSpec1 = 'Correcting labels for %s'
    formatSpec2 = '%s: %g / %g (%.2f %%)'
    
    wdir = Path(foldername)
    src = wdir / labelsource
    clss_file = src / "classes.txt"
    clss_file_new = wdir / "classes.txt"
    clss_new = []
    print(formatSpec1 %(wdir.name))
    if clss_file.is_file():
        clss = {i: s.strip() for i, s in enumerate(clss_file.read_text(encoding="utf-8").splitlines())}
        clss_lut = np.array([clss[i] for i in range(len(clss))], dtype=object)
        label_files = [p for p in src.glob("*.txt") if p.name != "classes.txt"]
    
        # Mapping from old class/index to new class/index.
        name2new = {}
        count = 0
        nf = len(label_files)
        for f in label_files:
            count = count + 1
            print(formatSpec2 %(wdir.name,count,nf, count/nf*100))
            if f.stat().st_size == 0: continue
            arr = np.loadtxt(f, ndmin=2)                                       # Loading label data.
            idx = arr[:, 0].astype(int)                                        # Class index.
            names = clss_lut[idx]                                              # Class name of the index.
            
            u_names, inv, first_pos = \
                np.unique(names, return_inverse=True, return_index=True) 
            order = np.argsort(inv)
            u_names_enc = u_names[order]                                       # Unique names in original order.
            
            known = np.array(list(name2new.keys()), dtype=object) \
                if name2new else np.array([], dtype=object)                    # Existing class names.
            new_names = u_names_enc[np.isin(u_names_enc, known, invert=True)]  # New names.
            start_id = len(clss_new)                                           # Starting index of new names.
            if new_names.size:
                name2new.update({n: start_id + j for j, n in enumerate(new_names)})
                clss_new.extend(new_names.tolist())
    
            arr[:, 0] = np.array([name2new[n] for n in names], dtype=int)      # Map index from old names to new names.
        
            fmt = ["%d"] + ["%.6f"] * (arr.shape[1] - 1)
            np.savetxt(wdir/f.name, arr, fmt=fmt)                                        
        clss_file_new.write_text("\n".join(clss_new) + "\n", encoding="utf-8")
    print('')
    
    return clss_new

#%% Correcting multiple folders within in a uder-defined folder.              #
def labelCorrectionLevel2(foldername, labelsource='labels_org', excludes=None):
    foldername = Path(foldername)
    if excludes is None:
        excludes_lower = []
    elif isinstance(excludes, (list, tuple, set)):
        excludes_lower = [e.lower() for e in excludes]
    else:
        excludes_lower = [excludes.lower()]

    # select subfolders excluding those in excludes_lower
    folders = [
        f for f in foldername.iterdir()
        if f.is_dir() and f.name.lower() not in excludes_lower
    ]

    clss = []
    for folder in folders:
        clssi = labelCorrectionLevel1(folder,labelsource,excludes=None)
        clss.append(clssi)
    return clss

#%% Modify Labelme or YOLO segementation json file class name and image data. #
def jsonCorrectionLevel1(foldername,jsonsource='segments_org',
                         classmap={'':''},overwrite=False):
    
    formatSpec1 = 'Correcting labels for %s'
    formatSpec2 = '%s: %g / %g, %s (%.2f %%), file existed, skipping...'
    formatSpec3 = '%s: %g / %g, %s (%.2f %%)'
    
    wdir = Path(foldername)
    src = wdir / jsonsource
    print(formatSpec1 %(wdir.name))
    label_files = natsorted([p for p in src.glob("*.json")])
    
    # Mapping from old class/index to new class/index.
    count = 0
    nf = len(label_files)
    clss_new = []
    
    for jf in label_files:
        jf_new = wdir/jf.name
        count = count + 1
        if jf_new.is_file() and not overwrite:
            print(formatSpec2 %(wdir.name, count,nf, jf.name, count/nf*100))
        else:
            print(formatSpec3 %(wdir.name, count,nf, jf.name, count/nf*100))
            if jf.stat().st_size == 0: continue
        
            # Loading json data.
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Removing imageData to reduce file size.
            for key in ("imageData", "img_data"):
                if key in data: data[key] = None
                
            # For labelme style.
            if "shapes" in data:
                for shape in data["shapes"]:
                    old_label = shape["label"]
                    new_label = old_label
                    if old_label in classmap and isinstance(classmap[old_label], str):
                        new_label = classmap[old_label]
                        shape["label"] = new_label
                    clss_new.append([old_label, new_label])
                    
            # For YOLO style
            if "annotations" in data:
                for ann in data["annotations"]:
                    old_id = ann.get("category_id")
                    new_id = old_id
                    if old_id in classmap and isinstance(classmap[old_id], int):
                        new_id = classmap[old_id]
                        ann["category_id"] = new_id
                    clss_new.append([old_label, new_label])
                    
            # Write the new data to disk.
            with open(jf_new, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)  
                
    # Get unique classes.
    clss_new = list({tuple(pair) for pair in clss_new})
    clss_new = np.array(clss_new, dtype=object)      
         
    print('')
    return clss_new

#%% Converting class id, flattered pixel xy, and W,H into YOLO label.         #
def bbox_from(cid,flat, W, H):
    xs, ys = flat[0::2], flat[1::2]
    x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
    xmn,xmx,ymn,ymx = x1,x2,y1,y2
    w,h = (xmx-xmn)/W, (ymx-ymn)/H
    xc,yc = (xmn+xmx)/(2*W), (ymn+ymx)/(2*H)
    return [cid,xc,yc,w,h]

#%% Converting json generated by labelme to ultralytics YOLO format.          #
def json2yolo(foldername,jsonsource='',task='segment',outfolder='',
              circlepointnumber=24,minimumpoints=3):
    
    formatSpec1 = 'Converting json to YOLO labels for %s'
    formatSpec2 = '%s: %g / %g (%.2f %%)'
    print(formatSpec1 % (Path(foldername).name))
    
    task = task.lower()
    wdir = Path(foldername)
    basename = wdir.name
    src = wdir / jsonsource
    outdir= wdir / outfolder
    
    clss_file= outdir / 'classes.txt'
    yaml_file= outdir / f'{basename}.yaml'
    clss_new = []
    json_files = [p for p in src.glob("*.json") if p.name.lower() != "annotations.json"]
    nf = len(json_files)
    outdir.mkdir(parents=True, exist_ok=True)
    
    name2new = {}
    count = 0
    for jp in json_files:
        count += 1
        print(formatSpec2 % (wdir.name, count, nf, count/nf*100.0))
        jd = json.loads(jp.read_text(encoding="utf-8"))
        W, H = jd.get("imageWidth"), jd.get("imageHeight")
        if not (isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0):
            continue
        
        image_name = jd.get("imagePath") or (jp.stem + ".jpg")
        out_txt = outdir / (Path(image_name).stem + ".txt")
        
        shapes = jd.get("shapes", []) or []
        names_in_file = [str(sh.get("label","")).strip() for sh in shapes if sh.get("label")]
        
        if names_in_file:
            # Identify classes not already included in clss_new.
            u_names, inv, first_pos = np.unique(
                np.array(names_in_file, dtype=object),
                return_inverse=True, return_index=True)
            order = np.argsort(inv)
            u_names_enc = u_names[order]                                   # Unique new class name.
            known = np.array(list(name2new.keys()), dtype=object) if name2new else np.array([], dtype=object)
            new_names = u_names_enc[np.isin(u_names_enc, known, invert=True)]
            if new_names.size:
                start_id = len(clss_new)
                for j, n in enumerate(new_names):
                    name2new[n] = start_id + j
                clss_new.extend(new_names.tolist())
                
        # Converting shape data to YOLO format.
        lines_out = []
        for sh in shapes:
            label = str(sh.get("label", "")).strip()
            if not label or label not in name2new:
                continue
            cid = name2new[label]
            st = (sh.get("shape_type") or "polygon").lower()
            pts = sh.get("points", [])

            # flatten to [x1,y1,x2,y2,...]
            flat = []
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    flat.extend([float(p[0]), float(p[1])])

            if st == "rectangle" and len(flat) >= 4:
                lines_out.append(bbox_from(cid, flat, W, H))

            elif st == "circle" and len(flat) >= 4:
                cx, cy, px, py = flat[0], flat[1], flat[2], flat[3]
                r = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                if task == "detect":
                    lines_out.append(bbox_from(cid, [cx - r, cy - r, cx + r, cy + r], W, H))
                else:
                    n = max(circlepointnumber, minimumpoints)
                    poly = []
                    for i in range(n):
                        ang = 2 * math.pi * i / n
                        poly.extend([cx + r * math.cos(ang), cy + r * math.sin(ang)])
                    norm = np.array(poly, dtype=float)
                    norm[0::2] /= W
                    norm[1::2] /= H
                    lines_out.append([cid] + norm.tolist())

            else:
                # polygon/line
                if task == "detect" and len(flat) >= 4:
                    lines_out.append(bbox_from(cid, flat, W, H))
                elif task in ("segment", "seg") and len(flat) >= 2 * minimumpoints:
                    norm = np.array(flat, dtype=float)
                    norm[0::2] /= W
                    norm[1::2] /= H
                    lines_out.append([cid] + norm.tolist())
                elif task == "auto":
                    if len(flat) >= 2 * minimumpoints:
                        norm = np.array(flat, dtype=float)
                        norm[0::2] /= W
                        norm[1::2] /= H
                        lines_out.append([cid] + norm.tolist())
                    elif len(flat) >= 4:
                        lines_out.append(bbox_from(cid, flat, W, H))
                        
        if lines_out:
            out_txt.unlink(missing_ok=True)
            with open(out_txt, "w", encoding="utf-8") as f:
                for row in lines_out:
                    f.write(" ".join(str(int(row[0])) if i==0 else f"{v:.6f}"
                                     for i,v in enumerate(row)) + "\n")
                    
    # Write class to .txt and .yaml format.
    if clss_new:
        clss_file.write_text("\n".join(clss_new) + "\n", encoding="utf-8") 
        yaml_text = (
            "# Train/val/test sets\n"
            "path:\n"
            "train: images/train\n"
            "val: images/val\n"
            "test:\n"
            "\n"
            "#Classes\n"
            "names:\n"
            ""
        )
        for n in clss_new: yaml_text += f"  {name2new[n]}: {n}\n"
        yaml_file.unlink(missing_ok=True)
        yaml_file.write_text(yaml_text, encoding="utf-8")
            
    return name2new

#%% choose split with biggest size deficit.                                   #
def best_split_for(stem, y, desired, current, current_total, target_total, rng):
    cs = y[stem]
    if not cs:
        nmax = max(
            current_total.keys(), 
            key=lambda s: (target_total[s] - current_total[s], rng.random())
            )
        return nmax
    scores = {}
    for s in ("train", "val", "test"):
        class_need = sum(max(0, desired[c][s] - current[c][s]) for c in cs)
        size_need = max(0, target_total[s] - current_total[s])
        scores[s] = (class_need, size_need, rng.random())
    return max(scores, key=lambda s: scores[s])

#%% Find an image file by stem in images_dir.                                 #
def find_image(stem, images_dir, img_exts):
    for ext in img_exts:
        p = images_dir / ("%s%s" % (stem, ext))
        if p.exists():
            return p
    cands = list(images_dir.glob(stem + ".*"))
    return cands[0] if cands else None

#%% Find an image file by stem in images_dir.                                 #
def place(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(src)
        except Exception:
            dst.symlink_to(src)
    else:
        raise ValueError("mode must be copy|move|link")

#%% Split integer n across keys in ratios (values sum ~1) so totals add to n. #
# Uses floor + distribute remaining to largest fractional parts.              #
def apportion_counts(n, ratios: dict):
    if n <= 0:
        return {k: 0 for k in ratios}
    raw  = {k: n * r for k, r in ratios.items()}
    base = {k: math.floor(v) for k, v in raw.items()}
    left = n - sum(base.values())
    order = sorted(ratios.keys(), key=lambda k: (raw[k] - base[k], k), reverse=True)
    for k in order[:left]:
        base[k] += 1
    return base

#%% Splitting a dataset into train/val/test based on given ratios.            %
def labelSplit(labeldir,imagedir='',outdir='anthro',ratios=(0.7,0.1,0.2),
               imgexts= ('.jpg','.jpeg','.png'),mode='copy',seed=42):
    
    # Screen print infomration control.
    fmt0   = 'Splitting data in %s'
    fmt1   = 'Target splitting ratio: %g%% train, %g%% val, %g%% test'
    fmt2   = 'Total ratio is %g%%, not 100 %%, skipping...'
    fmt3   = "\nSplit summary for images:"
    #fmt4   = "Total number: %d:"
    fmt5   = "%-5s: %3d /%4d (%5.2f%%) images (target ≈ %3d, %5.2f%%)"
    fmt6   = "\nSplit summary for instances:"
    #fmt7   = "Total number: %d:"
    fmt8   = "%-5s: %3d /%4d (%5.2f%%) instances"
    fmt9   = "\nPer-class image counts (actual vs desired):"
    fmt10  = "class %2d: train = %3d (≈%3d), val = %2d (≈%2d), test = %2d (≈%2d), %s"
    fmt11  = "[WARN] %d images missing under %s. First few stems: %s"
    fmt12  = "\nWriting %s.yaml to %s"
    
    #% Paths / setup
    print(fmt0 %(Path(labeldir).name))
    labeldir = Path(labeldir)
    imgdir = labeldir if imagedir == '' else labeldir / imagedir
    outdir = labeldir / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    
    train_r, val_r, test_r = ratios
    print(fmt1 %(train_r*100,val_r*100,test_r*100))
    sr = train_r + val_r + test_r
    cnt = {}
    if sr !=1 : print(fmt2 %(sr*100)); return

    rng = random.Random(seed)
    _, stems, _, ispvs = getFiles(imgdir, pext=list(imgexts))
    stems = [s for s, ipv in zip(stems, ispvs) if ipv == 0]
    
    # Building labels mapping from stem to set of classes.
    y = {}                                                                     # All class for each photo. 
    inst_per_stem = {}
    all_classes = set()
    for stem in stems:
        cls_set = set()
        ninst = 0
        lf = labeldir / ("%s.txt" % stem)
        txt = lf.read_text(encoding="utf-8").strip()
        for line in txt.splitlines():
            parts = line.strip().split()
            cid = int(float(parts[0]))
            cls_set.add(cid)
            ninst += 1
            
        y[stem] = cls_set
        inst_per_stem[stem] = ninst
        all_classes |= cls_set
        
    # Loading class names (fallback if missing)
    classes_file = labeldir / "classes.txt"
    classes_txt = classes_file.read_text(encoding="utf-8")
    names = [s.strip() for s in classes_txt.splitlines() if s.strip()]
    
    # Calculating target split.
    nd = len(stems)
    ntrain = max(1,round(nd * train_r))
    nval   = max(1,round(nd * val_r))
    ntest  = nd - ntrain - nval
    target_total = {'train': ntrain, 'val': nval, 'test': ntest}
    
    # All class list of each stem (photo).
    class_to_stems = defaultdict(list)
    for stem in stems:
        for c in y[stem]:
            class_to_stems[c].append(stem)
            
    # Desired per-class images per split.
    ratios_map = {'train': train_r, 'val': val_r, 'test': test_r}
    desired = {}
    for c in all_classes:
        m = len(class_to_stems[c])
        if m == 1:
            desired[c] = {'train': 1, 'val': 0, 'test': 0}
        elif m == 2:
            desired[c] = {'train': 1, 'val': 1, 'test': 0}
        elif m == 3:
            desired[c] = {'train': 1, 'val': 1, 'test': 1}
        else:
            des = apportion_counts(m, ratios_map)  # proportional base
            # ensure at least one goes to val
            if des['val'] < 1:
                donor = 'test' if des['test'] > 0 else 'train'
                des[donor] -= 1
                des['val']  += 1
            # ensure train >= val
            if des['train'] < des['val']:
                if des['test'] > 0:
                    des['test'] -= 1
                    des['train'] += 1
                else:
                    des['val']  -= 1
                    des['train'] += 1
            desired[c] = {'train': des['train'], 'val': des['val'], 'test': des['test']}
    
    # Current tallies
    current = {c: {'train': 0, 'val': 0, 'test': 0} for c in all_classes}
    current_total = {'train': 0, 'val': 0, 'test': 0}

    # Assignments
    assign = {}
    unassigned = set(stems)
    
    # Rare-to-common classes first.
    rare_to_common = sorted(all_classes, key=lambda c: len(class_to_stems[c]))
    for c in rare_to_common:
        for stem in class_to_stems[c]:
            if stem not in unassigned: continue
            s = best_split_for(stem, y, desired, current, current_total, target_total, rng)
            
            # If chosen split doesn't need this class but others do, try a better one
            needs = {ss: max(0, desired[c][ss] - current[c][ss]) 
                     for ss in ('train', 'val', 'test')}
            if needs[s] == 0 and any(needs.values()):
                s_alt = max(('train', 'val', 'test'),
                    key=lambda ss: (
                        needs[ss], 
                        max(0, target_total[ss] - current_total[ss]), 
                        rng.random())
                    )
                if needs[s_alt] > 0: s = s_alt
                
            assign[stem] = s
            unassigned.remove(stem)
            current_total[s] += 1
            for cc in y[stem]: current[cc][s] += 1
            
    # Assign remaining (including images with empty/no labels)
    rest = list(unassigned)
    rng.shuffle(rest)
    for stem in rest:
        def score(ss):
            size_def = target_total[ss] - current_total[ss]
            cls_def = sum(
                max(0, desired.get(c, {}).get(ss, 0) - current.get(c, {}).get(ss, 0)) 
                for c in y[stem]
                )
            return (size_def, cls_def, rng.random())
        s = max(('train','val','test'), key=score)
        assign[stem] = s
        current_total[s] += 1
        for cc in y[stem]: current[cc][s] += 1
        
    # Preparing folders
    for s in ('train','val','test'):
        (outdir / 'images' / s).mkdir(parents=True, exist_ok=True)
        (outdir / 'labels' / s).mkdir(parents=True, exist_ok=True)
        
    # Move/copy/link files
    missing_images = []
    for stem in stems:
        s = assign[stem]
        lf = labeldir / ("%s.txt" % stem)
        dst_lbl = outdir / 'labels' / s / ("%s.txt" % stem)
        if lf.exists():
            place(lf, dst_lbl, mode)
        else:
            dst_lbl.write_text("", encoding="utf-8")
            
        img = find_image(stem, imgdir, imgexts)
        if img is not None:
            place(img, outdir / 'images' / s / img.name, mode)
        else:
            missing_images.append(stem)
            
    # Report for overall splitting for photos.
    print(fmt3)
    #print(fmt4 %(nd))
    for s in ('train','val','test'):
        cnt[s] = sum(1 for st in stems if assign[st] == s)
        cnd = target_total[s]
        print(fmt5 % (s, cnt[s],nd, cnt[s]/nd*100, cnd, cnd/nd*100))
        
    # Report overall splitting for instances:
    inst_total = {'train': 0, 'val': 0, 'test': 0}
    for st in stems:
        s = assign[st]
        inst_total[s] += inst_per_stem.get(st, 0)
    nti = sum(inst_total.values())
    
    print(fmt6)
    #print(fmt7 %(total_instances))
    for s in ('train','val','test'):
        pct = (inst_total[s] / nti * 100.0) if nti else 0.0
        print(fmt8 % (s, inst_total[s], nti, pct))
    
    # Report for each class.
    print(fmt9)
    for c in sorted(all_classes):
        a_tr, a_va, a_te = current[c]['train'], current[c]['val'], current[c]['test']
        d_tr, d_va, d_te = desired[c]['train'], desired[c]['val'], desired[c]['test']
        print(fmt10 % (c, a_tr, d_tr, a_va, d_va, a_te, d_te, names[c]))
    
    # Report for images not found.
    if missing_images:
        preview = ", ".join(missing_images[:10]) + (" ..." if len(missing_images) > 10 else "")
        print(fmt11 % (len(missing_images), imgdir, preview))
        
    # Writing class to yaml.
    yaml_path = outdir / ("%s.yaml" % outdir.name)
    yaml_lines = [
        "# Train/val/test sets\n",
        "path:\n",
        "train: images/train\n",
        "val: images/val\n",
        "test:",
        "\n\n",
        "#Classes\n",
        "names:\n",
        ""]
    if cnt['test']>0: yaml_lines[4] = "test: images/test"  # newline fix
    
    for i, n in enumerate(names): yaml_lines.append("  %d: %s\n" % (i, n))
    yaml_path.write_text(''.join(yaml_lines) + "\n", encoding="utf-8")
    print(fmt12 % (outdir.name, yaml_path))
    
    return cnt

#%% Converting field observation data from the literature to cdf for plots.   #
def field2cdf(workdir,keyword='field',outkeyword='converted', \
              authorkeys = ['Buscombe2020', 'Mair2024']):
    
    outname = keyword + '_' + outkeyword
    fieldname = os.path.join(workdir, keyword)
    fieldnameout = os.path.join(workdir, outname)
    
    files, stems, ext, ispv = getFiles(workdir)
    files = [f for f, m in zip(files, ispv) if m == 0]
    stems = [s for s, m in zip(stems, ispv) if m == 0]
    
    # Build lookup sets for fast membership tests
    files_noext = {os.path.splitext(os.path.basename(f))[0] for f in files}
    stems_set   = set(map(str, stems))
    
    fout = pd.DataFrame()
    if os.path.isdir(fieldname):
        os.makedirs(fieldnameout, exist_ok=True)
        for author in authorkeys:
            fieldfile = os.path.join(fieldname, author + '.csv')
            if not os.path.isfile(fieldfile):
                continue
            fdata = pd.read_csv(fieldfile, index_col=False)
            
            if author.lower() == 'mair2024':
                keys = ['a_meters','b_meters','c_meters',
                        'a_pixels','b_pixels','c_pixels',
                        'w_meters','h_meters','r_meters',
                        'w_pixels','h_pixels','r_pixels',
                        'weight_kilograms','weight_grams']
                pairs = [
                    ('a_meters', 'b_meters'),
                    ('a_pixels', 'b_pixels'),
                    ('w_meters', 'h_meters'),
                    ('w_pixels', 'h_pixels')]
                
                site_base = fdata['site_id'].astype(str).\
                apply(lambda s: os.path.splitext(os.path.basename(s))[0])
                fdata = fdata.assign(_SiteBase=site_base)
    
                # Unique normalized site names
                sites = fdata['_SiteBase'].unique()
                for site in sites:
                    # require in either stems or files (by basename)
                    if (site in stems_set) or (site in files_noext):
                        fdata_st = fdata.loc[fdata["_SiteBase"] == site].copy()
                        fdata_st = fdata_st.replace(-9999, np.nan)      
                        fdata_st = fdata_st.dropna(axis=1, how="all")
                        fdata_st = fdata_st.dropna()
                        ng = fdata_st.shape[0]
                        cnames = fdata_st.columns
                        for cname in cnames:
                            if cname in keys:
                                y = np.asarray(fdata_st[cname])
                                f, x, _ = ecdfm(y)
                                
                                # Checking data for area-weight distribution.
                                area = None
                                for c1, c2 in pairs:
                                    if c1 in cnames and c2 in cnames:
                                        area = np.asarray(fdata_st[c1]) * np.asarray(fdata_st[c2])
                                        
                                if area is None:
                                    fout = pd.DataFrame({
                                        'grain_number': ng,
                                        "size_meters": np.asarray(x),
                                        "percent_finer_count": np.asarray(f) * 100.0
                                    })
                                else:
                                    yy = np.column_stack((y, area)) 
                                    fa, xa, _ = ecdfm(yy,option='byarea')
                                    fout = pd.DataFrame({
                                        'grain_number': ng,
                                        "size_meters": np.asarray(x),
                                        "percent_finer_count": np.asarray(f) * 100.0,
                                        "percent_finer_area": np.asarray(fa) * 100.0

                                    })
                                outname = site + '_' + cname + '.csv'
                                outfile = os.path.join(fieldnameout, outname)
                                fout.to_csv(outfile, index=False)
                                
            elif author.lower() == 'buscombe2020':
                cols = list(fdata.columns)
                img_col = 'files' if 'files' in cols else cols[0]
                value_cols = [c for c in cols if c != img_col]
    
                # Convert all value columns to numeric (coerce errors to NaN)
                fdata[value_cols] = fdata[value_cols].apply(
                    lambda s: pd.to_numeric(s, errors='coerce')
                )
    
                # Try to infer percentiles from column names; fallback to fixed list
                # e.g., columns like 'p5','P50','D84','95' -> extract 5,50,84,95
                def _extract_pct(c):
                    cs = str(c).strip()
                    # keep digits and at most one dot
                    import re
                    m = re.findall(r'[\d.]+', cs)
                    return float(m[0]) if m else None
    
                x = []
                auto = True
                for c in value_cols:
                    v = _extract_pct(c)
                    if v is None:
                        auto = False
                        break
                    x.append(v)
    
                if not auto or len(x) != len(value_cols):
                    x = [5, 10, 16, 25, 50, 75, 84, 90, 95]
                    if len(value_cols) != len(x):
                        x = list(range(1, len(value_cols) + 1))
    
                x = np.asarray(x, dtype=float)
                ng = x.shape[0]
    
                # Quick lookups
                files_basenames = {os.path.basename(p) for p in files}
                files_noext     = {os.path.splitext(os.path.basename(p))[0] for p in files}
    
                saved, skipped = 0, 0
                for _, row in fdata.iterrows():
                    imgname = str(row[img_col])
                    base = os.path.basename(imgname)
                    base_noext = os.path.splitext(base)[0]
    
                    # Only save if this image exists in your discovered files
                    if (base in files_basenames) or (base_noext in files_noext) or (base_noext in stems_set):
                        y = row[value_cols].to_numpy(dtype=float, copy=False)
                        # Drop NaNs pairwise with x
                        mask = np.isfinite(y)
                        if not mask.any():
                            skipped += 1
                            continue
                        xx = x[mask][:len(y[mask])]
                        yy = y[mask]
    
                        dfout = pd.DataFrame({
                            'grain_number': ng,
                            'size_pixels': yy,
                            'percent_finer_count': xx
    
                        })
    
                        outcsv = os.path.join(fieldnameout, f'{base_noext}_b_pixels.csv')
                        dfout.to_csv(outcsv, index=False)
                        saved += 1
                    else:
                        skipped += 1
    return fout
                     
#%%