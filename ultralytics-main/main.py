from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # model.load('yolo11n.pt')
    model.train(data='./datasets/three/data.yaml',
                imgsz=640,
                epochs=300,
                batch=64,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )

    # 评估模型在验证集上的性能
    metrics = model.val()

    # 在图像上执行对象检测
    results = model("./datasets/three/valid/images")
    for i in range(len(results)):
        results[i].save(f'./runs/output/{i}.jpg')

    # 将模型导出为 ONNX 格式
    path = model.export(format="onnx")  # 返回导出模型的路径