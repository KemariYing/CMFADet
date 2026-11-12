import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# absolute paths -> the script can be started from any working directory
ROOT = Path(__file__).resolve().parent

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = str(ROOT / 'runs/DV/train/CMFADet-640/weights/best.pt')
    model = YOLO(model_path)
    result = model.val( 
                        data=str(ROOT / 'dataset/data_DV_Multimodel.yaml'), # DATASET PATH
                        split='test', # train、val、test
                        imgsz=640,
                        batch=1,
                        iou=0.7,
                        # rect=False, 
                        # save_json=True, # if you need to cal coco metrice
                        save_conf=True, 
                        save_txt=True, # if you need to cal coco metrice
                        project=str(ROOT / 'runs/test/DV'),
                        name='CMFADet-640',
                        )
    
    # NOTE: the upstream script only handled task == 'detection'. For the OBB model
    # (OBBValidator -> OBBMetrics) `result.box` is available as well, so allow it here.
    if model.task in {'detection', 'obb'}:
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        # `result.box.p/r/f1/ap50/...` only contain the classes that were actually
        # evaluated (result.box.ap_class_index), and all_ap/f1 may be a plain list when
        # nothing was detected. Iterating over every name would raise IndexError, so
        # follow ap_class_index and normalise the containers first.
        ap_all = np.asarray(result.box.all_ap, dtype=float).reshape(-1, 10) if len(result.box.all_ap) else np.zeros((0, 10))
        f1_all = np.asarray(result.box.f1, dtype=float)
        mAP75_all = float(ap_all[:, 5].mean()) if ap_all.shape[0] else 0.0
        f1_mean = float(f1_all.mean()) if f1_all.size else 0.0

        for i, c in enumerate(result.box.ap_class_index):
            model_metrice_table.add_row([
                                        model_names[c],
                                        f"{result.box.p[i]:.4f}",
                                        f"{result.box.r[i]:.4f}",
                                        f"{result.box.f1[i]:.4f}",
                                        f"{result.box.ap50[i]:.4f}",
                                        f"{ap_all[i, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[i]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)", 
                                    f"{result.results_dict['metrics/precision(B)']:.4f}", 
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{f1_mean:.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{mAP75_all:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
