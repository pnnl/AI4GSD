import sys
from pathlib import Path
CWD = Path(__file__).resolve().parent
sys.path.insert(0, str(CWD.parent))                                       
import AI4GSD

## Demo of code.
if __name__ == "__main__":
    #%% (0) Define working directory and print computer info.
    AI4GSD.pcinfo()
    foldername = '20250517_YC_SmartphoneParkTest'                              # Parent folder where you save photos/videos.
    WorkDir = CWD.parent
    PhotoDir = str(WorkDir / 'Demo_Data' / foldername)                         # AI4GSD use string path.
    
    #%% (1) Extracting or copying your videos/photos to desired folder.
    AI4GSD.getPhotos(PhotoDir,nframe=2000, overwrite=True)                    # Use once.
    
    #%% (2) Obtaining scales from AI for validation purpuse.
    PP = AI4GSD.parameters()
    PP.Directory = PhotoDir
    PP.OverWriteAll = True                                                     # Overwrite everything if True.
    PP.OverWriteScaleFile = False
    PP.OverWriteScaleRows = True
    PP.ScaleSource = 'AI'                                                      # Use AI option.
    PP.ModelName = 'YOLO11m.1280.20250825'                                     # Which AI to use.
    PP.ComputingSource = 'local'                                               # What computing resource to use.
    PP.ImageSizeFactor = 1                                                     # Do you need to user a larger image size for inference?
    PP.ConfidenceThresholdUser = 0.85                                          # Minimum confidence threshold.
    PP.YOLOSave = False                                                        # Need to save AI prediction to folder.
    scales =  AI4GSD.metadata(PP)
    
    #%% (3) Obtaining segmentation of anthropogenic objects.
    PP = AI4GSD.parameters()
    PP.Directory = PhotoDir
    PP.FolderLevel = 'L1'
    PP.ModelName = 'YOLO11m.640.20250906'
    PP.ObjectName = 'anthro'
    PP.OverWriteSegementation = False
    PP.YOLOSave = False
    PP.YOLOIOU = 0.45
    PP.ComputingSource = 'local'
    PP.ImageSizeType = 'default'
    PP.ImageSizeFactor=1
    PP.SegmentationClassName = ['board','redbag','people']                     # What object class need to track.
    AI4GSD.background(PP)
    
    #%% (4) Obtaining grain size data.
    PP = AI4GSD.parameters()
    PP.Directory = PhotoDir
    PP.ModelName = 'YOLO11m.1280.20250322'
    PP.ImageSizeFactor = 1
    PP.LineThickness = 2
    PP.YOLOSave = False
    PP.SaveOverlayLabel = False
    PP.OverWriteAll = True                                                  
    PP.PrintOnScreen = True  
    PP.OverlayFontSize = 4
    PP.OverlayLegendFontSize = 4
    PP.OverlayGSDTickLineWidth = 4
    gsd = AI4GSD.Photo2GSD(PP)
    
    #%% (5) Applying quality control.
    gsdqc, stqc = AI4GSD.qualityControl(PhotoDir)
    
    #%% End.