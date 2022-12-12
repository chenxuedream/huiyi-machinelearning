## 绘意-机器学习

项目中所使用的机器学习部分代码，包括图像情绪分析/提取和音乐生成



### 图像情绪分析/提取

项目中使用ArtEmis项目，使用imagetoemotion模块训练实现。输入图片得到8八种情绪。八种情绪为 "敬畏", "娱乐","满足","兴奋","愤怒","厌恶","恐惧","悲伤",以及"其他"。



### 音乐生成

基于LSTM实现的音乐生成，数据集使用ABC version of the Nottingham Music Database。



**export.py**为将训练好的pt模型转换为onnx

**image2emotion.py**为使用python测试图像情绪提取的代码