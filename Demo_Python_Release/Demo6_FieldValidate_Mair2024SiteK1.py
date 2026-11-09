import os, sys
from pathlib import Path
CWD = Path(__file__).resolve().parent
sys.path.insert(0, str(CWD.parent))                                     
import AI4GSD

## Demo of code.
if __name__ == "__main__":
    #%% (0) Define working directory and print computer info.
    foldername = '20250916_YC_FieldValidateMair2024K1'  
    WorkDir = CWD.parent
    PhotoDir = str(WorkDir / 'Demo_Data' / foldername)                         # AI4GSD use string path.
    
    #%% (1) Extracting or copying your videos/photos to desired folder.
    AI4GSD.getPhotos(PhotoDir,nframe=2000, overwrite=True)                     # Use once.
    
    #%% (2) Obtaining scales from AI for validation purpuse.
    PP = AI4GSD.parameters()
    PP.Directory = PhotoDir
    PP.FolderLevel = 'L1'
    PP.OverWriteAll = False                                                    # Overwrite everything if True.
    PP.OverWriteScaleFile = True
    PP.OverWriteScaleRows = True
    PP.ScaleSource = 'default'                                                 # Use AI option.
    PP.ModelName = 'YOLO11m.1280.20250825'                                     # Which AI to use.
    PP.ComputingSource = 'local'                                               # What computing resource to use.
    PP.ImageSizeFactor = 1                                                     # Do you need to user a larger image size for inference?
    PP.YOLOSave = False                                                        # Need to save AI prediction to folder.
    PP.UserScaleFile = os.path.join(WorkDir,'Scales','Scales_V3_All.csv')
    PP.SiteID = 'K1'
    scales = AI4GSD.metadata(PP)
    
    #%% (3) Obtaining segmentation of anthropogenic objects.
    PP = AI4GSD.parameters()
    PP.Directory = PhotoDir
    PP.FolderLevel = 'L1'
    PP.ModelName = 'YOLO11m.640.20250527'
    PP.ObjectName = 'anthro'
    PP.OverWriteSegementation = True
    PP.YOLOSave = False
    PP.YOLOIOU = 0.45
    PP.ComputingSource = 'local'
    PP.ImageSizeType = 'default'
    PP.ImageSizeFactor=1
    #PP.SegmentationClassName = ['road','people','person']                     # What object class need to track.
    #AI4GSD.background(PP)
    
    #%% (4*) Obtaining minimum grain size and median size from field data.
    gsdname = 'K1_b_meters.csv'
    gsdcsv = os.path.join(PhotoDir,PP.FieldDataFolderName+'_'+PP.FieldDataConversionFolderName,gsdname)
    gsdf = AI4GSD.pd.read_csv(gsdcsv)
    d = gsdf['size_meters'].values
    p = gsdf['percent_finer_count'].values
    szmin = d.min()
    order = AI4GSD.np.argsort(p)
    ds = d[order]
    ps = p[order]
    d50 = AI4GSD.np.interp(50, ps, ds)
    
    #%% (4) Obtaining grain size data.
    PP = AI4GSD.parameters()
    PP.Directory = PhotoDir
    PP.ModelName = 'YOLO11m.1280.20250322'
    PP.ImageSizeFactor = 2
    PP.YOLOSave = False
    PP.SaveFigure = True
    PP.SaveOverlayAll = True
    PP.OverWriteAll = True                                                  
    PP.PrintOnScreen = True
    PP.GSDWeight = 'c'
    PP.GSDAxis = 'wh'
    PP.OverlayLableLineWidth = 1
    PP.SaveOverlayResolution = 1200
    PP.ConfidenceThresholdUser = 0.35
    PP.CutGrainSizeMethod = 'wh'
    PP.MinimumGrainSize = szmin
    PP.OverlayFontSizeScaleFactor = 10
    PP.SaveOverlayScaleDotSize = 6
    PP.OverlayLegendFontSize = 1.5
    PP.OverlayFontLineWidth = 0.5
    PP.OverlayGSDLineWidth = 0.5
    PP.OverlayD50MarkerSize = 0.8
    PP.OverlayFontBoundaryDistanceX = 0.01
    PP.OverlayFontBoundaryDistanceY = 0.01
    PP.OverlayFontColor = 'white'
    PP.OverlayFontColorForeground = 'black'
    PP.OverlayLegendLoc = 'best'
    PP.OverlayGSDMaxSize = 0.25
    PP.OverlayGSDXTickNumber = 6
    PP.OverlayGSDYTickNumber = 5
    PP.OverlayGSDColor = ['brown','blue','black','orange','cyan','red','purple']
    PP.OverlayGSDManualColor = ['brown','blue','black','orange','cyan','blue','purple']
    PP.OverlayGSDKey = ['AI4GSD: ','Manual','Field: ']
    PP.OverlayGSDLegend = ['w-c','h-c','r-c','area-width','area-height','area-diagonal']
    PP.OverlayGSDLegendField = ['b-c','b-c'] 
    PP.OverlayLegendAlpha = 0.6
    PP.OverlayLegendFrame = True
    PP.OverlayLegendFrame = True
    PP.OverlayLegendFaceColor = 'White'
    PP.OverlayLegendEdgeColor = 'none'
    gsd = AI4GSD.Photo2GSD(PP)
    
    #%% (4**) Computing relative error between model and field for D50.
    d50x = gsd.D50_x_count_meter.iloc[0]
    d50y = gsd.D50_y_count_meter.iloc[0]
    d50xy = (d50x + d50y)/2
    pe = (d50xy - d50)/d50*100
    print(f'Relative error between AI and field measurement is {pe:.2f}%')
    
    #%% (5) Applying quality control.
    gsdqc, stqc = AI4GSD.qualityControl(PhotoDir)
    
    #%% End.