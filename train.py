import warnings, os
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# absolute paths -> the script can be started from any working directory
ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    model = YOLO(str(ROOT / 'ultralytics/cfg/models/multimodal/Multi-SFEM-IRAFAB-CIF-obb-ATAH.yaml'))
    # model.load('yolo11n.pt') # loading pretrain weights
    # model = YOLO(str(ROOT / 'runs/DV/train/CMFADet-640/weights/last.pt')) ## Resume traing setting
    model.train(data=str(ROOT / 'dataset/data_DV_Multimodel.yaml'),
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=4, 
                # device='0,1',
                optimizer='SGD', # using SGD
                patience=300, # paper trains for the full 300 epochs (default 100 may early-stop)
                # resume=True, # Resume setting
                amp=False,
                # fraction=0.2,
                project=str(ROOT / 'runs/DV/train'),
                name='CMFADet-640',
                # val = False,
                )
