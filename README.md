 ## 我是素兮淑清，感谢我的队友Moses、fakenews，以及吉祥物CYYYYYY，我们一起获得了比赛的第二名
 
 
### **解决方案及算法**

**对于解决方案**
我们的主要思路还是比较常见的模型融合。只不过选择的模型可能会跟其他比赛常见的xgb、lgb、catboost以及一些深度模型有些区别。
我们主要构建了三个基于automl的模型，考虑到模型融合最好是具有差别的模型进行融合，因此我们对于这三个模型分别做了不同的特征工程或者不同的参数来保证模型的区别。这三个模型分别是**无特征工程的裸模型、基于一些特征等频或者等距分箱的模型以及分箱+分组特征+业务特征+robust标准化的模型**（特征工程较为简单，细节可见代码）。
当然我们在最后融合的时候也采取了一些不同的手段。我们进行了两层融合，第一层融合我们采取了两种不同的方式，分别为rank融合和加权融合，在第一层融合的基础上我们把rank融合和加权融合的结果再次进行rank融合，这就得到了最终线上0.97136的结果，长期赛线上分数为0.97142。

**对于算法**
与其说我们采用了某种算法，不如说我们采用了13个不同的模型做了三层到四层stacking，这13个模型包括knn及其变种、决策树及其变种、lgb及其变种、随机森林及其变种、xgb、catboost、nn及其变种。当然我们并不会耗费诸多精力去逐个构建13个甚至更多的基模型，并且逐个去做特征工程以及调参。在了解了亚马逊autogluon这个开源automl框架而且阅读了其官方文档之后，我们团队便采用这个开源框架来进行模型的训练，这样就减少了构建n个模型所耗费的时间和精力。

### **模型训练复现流程**

1.***关于复现，先打开code文件夹下的train.sh或者train.py，运行完后再打开code文件下的test.sh文件运行即可。若test.sh文件报错，可以直接运行test.py。【train.py中包含全部代码，运行完后可以在prediction_result中得到结果，若没有结果可运行test.sh或者test.py】***

2.user_data文件夹下model_data文件会在train.py运行结束后存放已经训练好的模型，这是project文件中占空间最大的文件。**可以打开查看，但是请勿删去其中文件**，因为test.py以及test.sh文件需要运行其中的文件。user_data文件夹下的其他文件是训练过程中产生的文件，也请不要删除。**模型运行时间极长，本机i5-12500h，rtx3060,16g内存（拯救者y9000p）保守估计要一天半时间，所以运行之前要做好心理准备。model_data文件夹在运行完之后会很大（所以没有上传），请运行train.py之前至少要保留30G的硬盘空间。**

3.系统为win10，ancaonda版本为2021.05 64位，python版本为3.8.8。必须的包为三个，一个是pandas，版本为1.4.3，另一个是autogluon，版本为0.5.0，最后一个是sklearn，版本为1.0.2。

**！！！！！！注意！！！！！！**
若运行环境为win10或者其他windows版本，在运行test.sh文件或者test.py文件时，需要将anaconda环境下的Lib/pathlib.py中代码大概1040行的函数进行如下替换：

#修改前
def __new__(cls, *args, **kwargs):
        if cls is Path:
            cls = WindowsPath if os.name == 'nt' else PosixPath
        self = cls._from_parts(args, init=False)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system"
                                      % (cls.__name__,))
        self._init()
        return self

#修改后    
def __new__(cls, *args, **kwargs):
        if cls is Path:
            # cls = WindowsPath if os.name == 'nt' else PosixPath
            cls = WindowsPath
        self = cls._from_parts(args, init=False)
        # Windows doesn't support PosixPath
        if type(self) == PosixPath:
            cls = WindowsPath
            self = cls._from_parts(args, init=False)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system"
                                      % (cls.__name__,))
        self._init()
        return self


**这样做的原因在于windows运行linux环境下跑出来的模型时会报错，因此需要改动。【若模型均在windows下运行则不需要替换】**

### **其他**

**若模型复现出现问题，可发邮件联系我看到后会回复，我的邮箱是pxduan0218@163.com。**
