任务：
给定一张输入图像，找到哪些位置存在零件，以及零件的类型。
零件的类型一共三种，分别为大中。大零件可能存在于9个位置，中零件12个位置，小零件15个位置，每张图像中只存在一种类型的零件。


要求：
1. Disks 文件夹存放所有图像。该目录下有三个子目录: Disks/1，Disks/2，Disks/3 ，分别包含小，中，大三种类型的图像. 
   此外，其中每个子文件夹中包含一个labels.txt文件，文件中第一行表示图像类型，例如，1，2，3，以下每一行表示一个文件以及其中包含的标签。

例如： 在Disks/2 目录下有3张图片image1.jpg, image2.jpg, image3.jpg, 那么labels.txt 内容可能如下。
注意！只使用空格作为分隔符，不要使用任何其他分隔符，切记不要使用tab。空格不要敲两下，每次只敲一个空格。

2
image1 1 2 3 4 5 6 7 8 9 10 11 12
image2 1 2 3 4 5 6 7 8 9 10 11 12
image3 3



2. template_bbx 文件夹存放所有标签对应的坐标，1.json, 2.json, 3.json 分别对应小，中和大零件。
    每个json文件由readme 软件生成，也可以自己手工写。
    用labelme软件的语法是
    labelme Disks --nodata --autosave
    1. 用该软件标注每一类图像中所有的坐标框，按control+R开始选择长方形。每个长方形尽可能框住一个位置的零件，同时避免框住其他位置的零件。
    2. 从左上到右下角框的长方形标签为n_m, 其中n=1,2,3 为零件类型，m=1....15 为位置. 例如，小零件左上角的框的标签为1_1, 大零件右下角的框的标签为3_9。
    3. 自动生成json文件，修改文件名后放到template_bbx文件夹中, 小零件命令为1.json，中零件为2.json, 大零件为3.json

文件示例可以查看template_bbx目录下标注好的json文件。

    

操作：
0.准备工作，切换到conda环境中
    conda activate torch2
1.首先用 DataProcessing 进行截图，所有的截图储存在 Crop文件夹中
    python DataProcess.py
2.训练模型
    python train.py
3.测试图像  Disks\2\image1.jpg
    python test.py Disks\2\image1.jpg

注意：可通过参数修改默认路径


软件升级：
软件代码托管在gitee上 https://gitee.com/Liszt8888/DiskRecog
在根目录下 conda环境中
    git pull https://gitee.com/Liszt8888/DiskRecog.git master



环境：
pytorch==1.13


算法逻辑：
1. DataProcess.py: 将所有的Disk目录下的图像按照template_bbx下json文件的指定范围进行切割，切割后的小图像块储存在Crop目录下.
2. train.py: 使用Crop目录下的图像训练模型，训练好的模型保存为当前目录下的 best.pt 文件.
3. test.py: 使用训练好的模型进行测试.