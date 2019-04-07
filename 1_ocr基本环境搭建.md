### 配置
前期需要安装anconda
在python3.5环境下配
按照setup-win-cpu.txt逐行在cmd下执行

### 模型文件  
链接：https://pan.baidu.com/s/1EIK-W3Z0Dp_CWK8FtQLivw 
提取码：jijl 

### 测试
将所需要的单个文件放入test文件夹下
执行python demo.py

### 出现的问题
* 1.执行python demo.py 出现找不到文件的问题：  
将路径改为以chinese-ocr根目录的相对路径
* 2.ctpn的配置  
参照ctpn中github页面的issue中的win cpu配置问题https://github.com/eragonruan/text-detection-ctpn/issues/73
* 3.unable-to-find-vcvarsall-bat问题  
python3.5配置，需要下载相关vs2015的相关文件，见页面https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/#comments
