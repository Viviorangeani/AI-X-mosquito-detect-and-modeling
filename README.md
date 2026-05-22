# AI-X-mosquito-detect-and-modeling
面向白纹伊蚊与玉米病斑的 AI 监测项目

数据集与可直接运行的代码：
corn：https://www.kaggle.com/code/viviorangeani/cornleaf-detect
mosquito：https://www.kaggle.com/code/viviorangeani/mosquitoalert

**目标检测模型（YOLOv8n）+数学建模应用**

我们的整体架构可以用四个层来概括：
1.前端感知层：YOLOv8n在边缘设备上实时检测，输出目标类别、经纬度、时间戳。
2.数据处理层：把经纬度映射到平面坐标，划分网格和时间段。
3.分析层：聚类分析+熵权法风险建模+马尔可夫链动态预测。
4.决策展示层：生成五级风险热力图，给出监控和防治建议。
