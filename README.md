## YOLOV3：You Only Look Once目标检测模型在Pytorch当中的实现

## 所需环境

torch == 1.2.0

## 权重文件下载

训练所需的yolo_weights.pth可以在百度云下载。  
链接: https://pan.baidu.com/s/1ncREw6Na9ycZptdxiVMApw   
提取码: appk

## 训练步骤

1. 本文使用VOC格式进行训练。  

2. 训练前将VOC2007和试管的标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  

3. 训练前将VOC2007和试管的图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  

4. 在训练前利用voc2yolo3.py文件生成对应的txt。  

5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
   
   ```python
   classes = ["tube", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
   ```

6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  

7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类**，示例如下：   
   model_data/new_classes.txt文件内容为：   
   
   ```python
   cat
   dog
   ...
   ```

8. **修改utils/config.py里面的classes，使其为要检测的类的个数**。   

9. 运行train.py即可开始训练。

# 

## Reference

https://github.com/qqwweee/keras-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch
