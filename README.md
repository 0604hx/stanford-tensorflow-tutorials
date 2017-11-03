# tf-stanford-tutorials
This repository contains code examples for the course CS 20SI: TensorFlow for Deep Learning Research. <br>
It will be updated as the class progresses. <br>
Detailed syllabus and lecture notes can be found here http://cs20si.stanford.edu

# Note (as of July 11, 2017)
I've updated the code to TensorFlow 1.2 and Python3, except the code for chatbot. I will update the code for chatbot soon.



## Models include: <br>
### In the folder "examples": <br>
Linear Regression with Chicago's Fire-Theft dataset<br>
Logistic Regression with MNIST<br>
Word2vec skip-gram model with NCE loss<br>
Convnets with MNIST<br>
Autoencoder (by Nishith Khandwala)<br>
Deepdream (by Jon Shlens)<br>
Character-level language modeling <br>
<br>
### In the folder "assignments":<br>
Style Transfer<br>
Chatbot using sequence to sequence with attention<br>
<br>
## Misc<br>
Examples on how to use data readers, TFRecord<br>
Embedding visualization with TensorBoard<br>
Usage of summary ops<br>
Exercises to be familiar with other special TensorFlow ops<br>
Demonstration of the danger of lazy loading <br>
Convolutional GRU (CRGU) (by Lukasz Kaiser)

## 附录

python version : `3.5+`

### 错误记录

**[OSX] python is not install as frameword**
```text
from matplotlib.backends import _macosx
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. 
```

just run 
```commandline
echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
```

enjoy!

**pip 安装 scipy 失败**

提示信息：`numpy.distutils.system_info.NotFoundError: no lapack/blas resources found`

此时从 [python extension](https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy) 下载与系统匹配的 `whl` 文件，然后执行：`pip install {whl 文件路径}` 即可完成安装