'''
验证 YOLO 模型的识别准确率，最直接、标准的方法是使用其官方库 Ultralytics 自带的验证（Val）模式。你只需要准备好标注好的验证集，一条命令就能得到精确率、召回率以及核心指标 mAP。

核心评估指标：看懂验证结果
运行验证后，会输出一系列指标，其中最关键的是 mAP（平均精度均值）。

mAP50：当预测框与真实框的重叠度（IoU）达到 0.5 时，就认为是检测正确。这是最常用的指标，数值越高，说明模型“找得准”的能力越强。

mAP50-95：在 IoU 从 0.5 到 0.95（步长 0.05）的多个阈值下计算 mAP 再取平均。这个指标更严格，要求预测框的位置非常精确，能更全面地反映模型定位的精准度。

Precision（精确率）与 Recall（召回率）：精确率指“检出的目标里有多少是真的”，召回率指“真实的目标里有多少被检出了”。两者通常需要权衡。

命令
/home/xiaozhang/miniconda3/envs/yolov8_env/bin/python /home/xiaozhang/code/ultralytics/util/val_model.py
'''

from ultralytics import YOLO

# 加载你的模型
model = YOLO('/home/xiaozhang/code/navigator_all/yolo_llm_code/model/best.pt')

# 开始验证，可以指定数据集
metrics = model.val(data='/home/xiaozhang/code/PictureAugment/data/data.yaml')

# 直接打印核心指标
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")