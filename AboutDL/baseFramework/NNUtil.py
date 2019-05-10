#-*- coding: UTF-8 -*-
import os
import math
import datetime
import unicodedata
import numpy as np
import tensorflow as tf
import scipy.stats as ss
import cx_Oracle

from scipy import interp
from sklearn import metrics


def init_w(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

#参数shape即为当前层的神经元个数
def init_b(shape, name):
    return tf.get_variable(name, shape, initializer=tf.zeros_initializer())

#定义先行映射过程，其中net是线性映射的输入，shape是其中权值矩阵的shape，appendix则是线性映射这一套运算步骤所属的name_scope的后缀
def fully_connected_linear(net, shape, appendix, pruner=None):
    with tf.name_scope("Linear{}".format(appendix)):
        w = init_w(shape, "W{}".format(appendix))#init_w初始化权值矩阵
        if pruner is not None:
            w = pruner.prune_w(*pruner.get_w_info(w))
        b = init_b(shape[1], "b{}".format(appendix))#init_b初始化偏置量
        return tf.add(tf.matmul(net, w), b, name="Linear{}".format(appendix))

#可视化日志路径：命令行tensorboard --logdir=D:\tmp\tbLogs --host=127.0.0.1
def prepare_tensorboard_verbose(sess):
    tb_log_folder = os.path.join(os.path.sep, "tmp", "tbLogs",
        str(datetime.datetime.now())[:19].replace(":", "-"))
    train_dir = os.path.join(tb_log_folder, "train")
    test_dir = os.path.join(tb_log_folder, "test")
    for tmp_dir in (train_dir, test_dir):
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
    tf.summary.merge_all()
    tf.summary.FileWriter(train_dir, sess.graph)

#定义能够根据模型预测（pred）与真值（y）来评估模型表现得类
class Metrics:
    """
    定义两个辅助字典
    sign_dict：key为metric名，value为±1，其中1说明该metric越大越好，-1说明该metric越小越好
    require_prob：key为metric名，value为True或False，其中true说明该metric需要接受一个概率预测值，false说明该metric需要接受一个类别预测值
    """
    sign_dict = {
        "f1_score": 1,
        "r2_score": 1,
        "auc": 1, "multi_auc": 1, "acc": 1, "binary_acc": 1,
        "mse": -1, "ber": -1, "log_loss": -1,
        "correlation": 1, "top_10_return": 1
    }
    require_prob = {key: False for key in sign_dict}
    require_prob["auc"] = True
    require_prob["multi_auc"] = True

    #定义能够调整向量形状以适应相应的metric函数的输入要求的方法
    #由于scikit-learn的metrics中定义的函数接收的参数都是一维数组，所以该方法的主要目的是把二维数组转为合乎要求的、相应的一维数组
    @staticmethod
    def check_shape(y, binary=False):
        y = np.asarray(y, np.float32)
        if len(y.shape) == 2:#当y是二维数组时
            if binary:#如果是二分类问题（y=0或1）
                if y.shape[1] == 2:#而且y还是二维数组的话，就返回第二列的相应预测值
                    return y[..., 1]
                assert y.shape[1] == 1
                return y.ravel()#否则就把y铺平后返回
            return np.argmax(y, axis=1)#如果不是二分类问题的话，返回y的argmax
        return y#当y不是二维数组（即y是一维数组）时，直接返回y即可
    
    #定义计算f1_score的方法，一般用于不均衡的二分类问题
    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(Metrics.check_shape(y), Metrics.check_shape(pred))

    #定义计算r2_score的方法，一般用于回归问题
    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y, pred)

    #定义计算auc的方法，也是一般用于不均衡的二分类问题
    @staticmethod
    def auc(y, pred):
        return metrics.roc_auc_score(Metrics.check_shape(y, True),Metrics.check_shape(pred, True))

    #定义计算多分类auc的方法，一般用于不均衡的多分类问题
    @staticmethod
    def multi_auc(y, pred):
        n_classes = pred.shape[1]
        if len(y.shape) == 1:
            y = Toolbox.get_one_hot(y, n_classes)
        fpr, tpr = [None] * n_classes, [None] * n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], pred[:, i])
        new_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        new_tpr = np.zeros_like(new_fpr)
        for i in range(n_classes):
            new_tpr += interp(new_fpr, fpr[i], tpr[i])
        new_tpr /= n_classes
        return metrics.auc(new_fpr, new_tpr)

    #定义计算准确率的方法，一般用于均衡的二分类问题与多分类问题
    @staticmethod
    def acc(y, pred):
        return np.mean(Metrics.check_shape(y) == Metrics.check_shape(pred))

    #定义计算二分类准确率的方法
    @staticmethod
    def binary_acc(y, pred):
        return np.mean((y > 0) == (pred > 0))

    #定义计算（欧氏）平均距离的方法，一般用于分类问题
    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y.ravel() - pred.ravel()))

    @staticmethod
    def ber(y, pred):
        mat = metrics.confusion_matrix(Metrics.check_shape(y), Metrics.check_shape(pred))
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return 0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))

    #定义计算log_loss的方法，一般用于分类问题
    @staticmethod
    def log_loss(y, pred):
        return metrics.log_loss(y, pred)

    #定义计算相关性系数的方法，一般用于回归问题
    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y, pred)[0])

    @staticmethod
    def top_10_return(y, pred):
        return np.mean(y[pred >= np.percentile(pred, 90)])

    @staticmethod
    def from_fpr_tpr(pos, fpr, tpr, metric):
        if metric == "ber":
            return 0.5 * (1 - tpr + fpr)
        return tpr * pos + (1 - fpr) * (1 - pos)

#定义能够根据模型预测（pred）与真值（y）来计算损失的类
class Losses:
    #定义（欧式）距离损失函数
    @staticmethod
    def mse(y, pred, _, weights=None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred, tf.reshape(weights, [-1, 1]))

    #定义交叉熵损失函数：注意需要根据pred是否已经是概率向量来调整损失的计算方法
    @staticmethod
    def cross_entropy(y, pred, already_prob, weights=None):
        if already_prob:#如果pred已经是概率向量的话，就对其取对数，从而后面计算Softmax+Cross Entropy时就能还原出pred本身
            eps = 1e-12
            pred = tf.log(tf.clip_by_value(pred, eps, 1 - eps))
        if weights is None:
            return tf.losses.softmax_cross_entropy(y, pred)
        return tf.losses.softmax_cross_entropy(y, pred, weights)
    
    #定义相关性系数损失函数
    @staticmethod
    def correlation(y, pred, _, weights=None):
        #利用tf.nn.moments算出均值与方差
        y_mean, y_var = tf.nn.moments(y, 0)
        pred_mean, pred_var = tf.nn.moments(pred, 0)
        #利用均值、方差与相应公式算出相关性系数
        if weights is None:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean))
        else:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean) * weights)
        #将损失设置为负相关性，从而期望模型输出与标签的相关性增加
        return -e / tf.sqrt(y_var * pred_var)

    @staticmethod
    def perceptron(y, pred, _, weights=None):
        if weights is None:
            return -tf.reduce_mean(y * pred)
        return -tf.reduce_mean(y * pred * weights)

    @staticmethod
    def clipped_perceptron(y, pred, _, weights=None):
        if weights is None:
            return -tf.reduce_mean(tf.maximum(0., y * pred))
        return -tf.reduce_mean(tf.maximum(0., y * pred * weights))

    @staticmethod
    def regression(y, pred, *_):
        return Losses.correlation(y, pred, *_)

#定义能够根据输入x返回激活值的类（提供激活函数的类）
class Activations:
    @staticmethod
    def elu(x, name):
        return tf.nn.elu(x, name)

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name)

    @staticmethod
    def selu(x, name):
        return tf.nn.selu(x,name)
        #alpha = 1.6732632423543772848170429916717
        #scale = 1.0507009873554804934193349852946
        #return tf.multiply(scale, tf.where(x >= 0., x, alpha * tf.nn.elu(x)), name)

    @staticmethod
    def sigmoid(x, name):
        return tf.nn.sigmoid(x, name)

    @staticmethod
    def tanh(x, name):
        return tf.nn.tanh(x, name)

    @staticmethod
    def softplus(x, name):
        return tf.nn.softplus(x, name)

    @staticmethod
    def softmax(x, name):
        return tf.nn.softmax(x, name=name)

    @staticmethod
    def sign(x, name):
        return tf.sign(x, name)

    @staticmethod
    def one_hot(x, name):
        return tf.multiply(x, tf.cast(tf.equal(x, tf.expand_dims(tf.reduce_max(x, 1), 1)), tf.float32), name=name)

#实现一些小工具（读取数据时可能用到）
class Toolbox:

    #判断一个字符串是否是数值型的字符串
    @staticmethod
    def is_number(s):
        try:
            s = float(s)
            if math.isnan(s):
                return False
            return True
        except ValueError:
            try:
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                return False

    @staticmethod
    def all_same(target):
        x = target[0]
        for new in target[1:]:
            if new != x:
                return False
        return True

    @staticmethod
    def all_unique(target):
        seen = set()
        return not any(x in seen or seen.add(x) for x in target)

    @staticmethod
    def warn_all_same(i, logger=None):
        warn_msg = "在列 {} 中的所有值都是相同的, 它们将被当作多余的".format(i)
        print(warn_msg) if logger is None else logger.debug(warn_msg)

    @staticmethod
    def warn_all_unique(i, logger=None):
        warn_msg = "在列 {} 中的所有值都是独一的, 它们将被当作多余的".format(i)
        print(warn_msg) if logger is None else logger.debug(warn_msg)

    @staticmethod
    def pop_nan(feat):
        no_nan_feat = []
        for f in feat:
            try:
                f = float(f)
                if math.isnan(f):
                    continue
                no_nan_feat.append(f)
            except ValueError:
                no_nan_feat.append(f)
        return no_nan_feat

    @staticmethod
    def shrink_nan(feat):
        new = np.asarray(feat, np.float32)
        new = new[~np.isnan(new)].tolist()
        if len(new) < len(feat):
            new.append(float("nan"))
        return new

    #定义能够从分隔符为sep的值的文件中读入数据的方法，include_header表示是否有头
    @staticmethod
    def get_data(file, sep=",", include_header=False, logger=None):
        msg = "获取数据中..."
        print(msg) if logger is None else logger.debug(msg)
        data = [[elem if elem else "nan" for elem in line.strip().split(sep)] for line in file]
        if include_header:
            return data[1:]
        return data

    @staticmethod
    def get_data_db(dbStr,user,password,sql):
        try:
            connection = cx_Oracle.connect(user, password, dbStr)#连接数据库
            cursor = connection.cursor()#获得操作游标
            cursor.execute(sql)
            all_data = cursor.fetchall()#获取全部数据，all_data是个list，每一行是个tuple
        finally:
            connection.close()
            return []
        return all_data

    #定义能够将y转化为独热表示的方法
    @staticmethod
    def get_one_hot(y, n_class):
        if y is None:
            return
        one_hot = np.zeros([len(y), n_class])
        one_hot[range(len(one_hot)), np.asarray(y, np.int)] = 1
        return one_hot
    
    #提取数据的信息，data：（目标）数据；is_regression：标识当前数据对应的问题是否是回归问题的参数
    @staticmethod
    def get_feature_info(data, numerical_idx, is_regression, logger=None):
        generate_numerical_idx = False
        if numerical_idx is None:#如果没有提供，就说明要用程序来生成
            generate_numerical_idx = True
            numerical_idx = [False] * len(data[0])
        else:
            numerical_idx = list(numerical_idx)
        #定义“数据的转置”，从而一个特征就对应着一行
        data_t = data.T if isinstance(data, np.ndarray) else list(zip(*data))
        if type(data[0][0]) is not str:#如果数据类型不是str则需要考虑其存在缺失值nan的情形：在Python中nan和nan之间会被视为不同的元素
            shrink_features = [Toolbox.shrink_nan(feat) for feat in data_t]
        else:
            shrink_features = data_t
        #利用Python自带的集合数据机构
        feature_sets = [ set() if idx is None or idx else set(shrink_feature)
            for idx, shrink_feature in zip(numerical_idx, shrink_features) ]
        n_features = [len(feature_set) for feature_set in feature_sets]
        all_num_idx = [True if not feature_set else all(Toolbox.is_number(str(feat)) for feat in feature_set)
            for feature_set in feature_sets]
        if generate_numerical_idx:#如果要生成numerical_idx的话，就先生成numpy数组版本的shrink_feature
            np_shrink_features = [shrink_feature if not all_num else np.asarray(shrink_feature, np.float32)
                for all_num, shrink_feature in zip(all_num_idx, shrink_features)]
            #然后根据它来看，是否存在特征类型为整数类型且特征取值各不相同的特征，存在的话说明可能是id所有为冗余特征
            all_unique_idx = [len(feature_set) == len(np_shrink_feature)
                and (not all_num or np.allclose(np_shrink_feature, np_shrink_feature.astype(np.int32)))
                for all_num, feature_set, np_shrink_feature in zip(all_num_idx, feature_sets, np_shrink_features)]
            numerical_idx = Toolbox.get_numerical_idx(feature_sets, all_num_idx, all_unique_idx, logger)
            for i, numerical in enumerate(numerical_idx):#如果有冗余特征，就将all_num_idx相应位置的元素置为None
                if numerical is None:
                    all_num_idx[i] = None
        else:#提供了numerical_idx
            for i, (feature_set, shrink_feature) in enumerate(zip(feature_sets, shrink_features)):
                if i == len(numerical_idx) - 1 or numerical_idx[i] is None:
                    continue
                if feature_set:#如果某个特征为离散型特征
                    if len(feature_set) == 1:#而且特征取值个数为1的话，说明对应特征为冗余特征
                        Toolbox.warn_all_same(i, logger)
                        all_num_idx[i] = numerical_idx[i] = None
                    continue
                if Toolbox.all_same(shrink_feature):#如果某个特征的所有特征都取同一个值的话，说明为冗余特征
                    Toolbox.warn_all_same(i, logger)
                    all_num_idx[i] = numerical_idx[i] = None
                elif numerical_idx[i]:#若该特征为整数型特征，且取值各不相同的话，说明很可能是id，所以也是冗余的
                    shrink_feature = np.asarray(shrink_feature, np.float32)
                    if np.max(shrink_feature[~np.isnan(shrink_feature)]) < 2 ** 30:
                        if np.allclose(shrink_feature, np.array(shrink_feature, np.int32)):
                            if Toolbox.all_unique(shrink_feature):
                                Toolbox.warn_all_unique(i, logger)
                                all_num_idx[i] = numerical_idx[i] = None
        if is_regression:#如果指定了是回归问题的话，就要更改一些自动生成的信息
            all_num_idx[-1] = numerical_idx[-1] = True
            feature_sets[-1] = set()
            n_features.pop()
        return feature_sets, n_features, all_num_idx, numerical_idx

    @staticmethod
    def get_numerical_idx(feature_sets, all_num_idx, all_unique_idx, logger=None):
        rs = []
        print("生成 numerical_idx 中")
        for i, (feat_set, all_num, all_unique) in enumerate(zip(feature_sets, all_num_idx, all_unique_idx)):
            if len(feat_set) == 1:
                Toolbox.warn_all_same(i, logger)
                rs.append(None)
                continue
            no_nan_feat = Toolbox.pop_nan(feat_set)
            if not all_num:
                if len(feat_set) == len(no_nan_feat):
                    rs.append(False)
                    continue
                if not all(Toolbox.is_number(str(feat)) for feat in no_nan_feat):
                    rs.append(False)
                    continue
            no_nan_feat = np.array(list(no_nan_feat), np.float32)
            int_no_nan_feat = no_nan_feat.astype(np.int32)
            n_feat, feat_min, feat_max = len(no_nan_feat), no_nan_feat.min(), no_nan_feat.max()
            if not np.allclose(no_nan_feat, int_no_nan_feat):
                rs.append(True)
                continue
            if all_unique:
                Toolbox.warn_all_unique(i, logger)
                rs.append(None)
                continue
            feat_min, feat_max = int(feat_min), int(feat_max)
            if np.allclose(np.sort(no_nan_feat), np.linspace(feat_min, feat_max, n_feat)):
                rs.append(False)
                continue
            if feat_min >= 20 and n_feat >= 20:
                rs.append(True)
            elif 1.5 * n_feat >= feat_max - feat_min:
                rs.append(False)
            else:
                rs.append(True)
        return np.array(rs)


#训练监控器：核心，半自动化的重要原因之一
class TrainMonitor:
    """
    初始化结构
    sign:模型所用的评价指标（metric）的“符号”，为1时意味着metric越大越好（比如准确率acc）为-1时意味着越小越好（比如均方误差mse）
    snapshot_ratio：“周期性检查是否需要进行参数备份”中的“周期”
    history_ratio：滑动窗口长度相对于上述“周期”的比率
    extension：每次延长的训练步数
    std_floor、std_ceiling：阈值和提高鲁棒性的参数
    tolerance_ratio：默认是2，避免轻易过拟合设置为更大的数
    """
    def __init__(self, sign, snapshot_ratio, history_ratio=3, tolerance_ratio=10, extension=5, std_floor=0.01, std_ceiling=0.01):
        self.sign = sign
        self.snapshot_ratio = snapshot_ratio
        self.n_history = int(snapshot_ratio * history_ratio)#记录滑动窗口长度的属性
        self.n_tolerance = int(snapshot_ratio * tolerance_ratio)#记录阈值的属性
        self.extension = extension
        self.std_floor, self.std_ceiling = std_floor, std_ceiling
        self._scores = []#记录model所有历史“分数”的属性
        self.flat_flag = False#记录是否开始对稳定程度计数的属性
        self._is_best = self._running_best = None#记录当前是否是历史最优表现得属性
        self._running_sum = self._running_square_sum = None#记录“滑动求和”，“滑动平方和”
        self._descend_increment = self.n_history * extension / 30#记录延长步数所带来的惩罚的属性

        self._over_fit_performance = math.inf#记录model刚陷入过拟合时得表现的属性
        self._best_checkpoint_performance = -math.inf
        self._descend_counter = self._flat_counter = self.over_fitting_flag = 0
        self.info = {"terminate": False, "save_checkpoint": False, "save_best": False, "info": None}

    def punish_extension(self):
        self._descend_counter += self._descend_increment

    def _update_running_info(self, last_score, n_history):
        #如果实际滑动窗口长度还没有达到预设的滑动窗口长度，又或是实际滑动窗口长度与已有score的总数恰好一致
        if n_history < self.n_history or n_history == len(self._scores):
            if self._running_sum is None or self._running_square_sum is None:#注意是跳过了第一次的，所以初始化时要把前两次的score一起算
                self._running_sum = self._scores[0] + self._scores[1]
                self._running_square_sum = self._scores[0] ** 2 + self._scores[1] ** 2
            else:#已经初始化了则直接加上最新的score即可
                self._running_sum += last_score
                self._running_square_sum += last_score ** 2
        else:
            previous = self._scores[-n_history - 1]
            self._running_sum += last_score - previous
            self._running_square_sum += last_score ** 2 - previous ** 2
        #如果还没初始化则初始化相应的属性，注意这里的improvement是“当前表现相对于历史最优表现的提升”
        #而且如果当前表现不是历史最优表现得话，improvement就会是0
        if self._running_best is None:
            if self._scores[0] > self._scores[1]:
                improvement = 0
                self._running_best, self._is_best = self._scores[0], False
            else:
                improvement = self._scores[1] - self._scores[0]
                self._running_best, self._is_best = self._scores[1], True
        elif self._running_best > last_score:#根据当前表现和历史最优表现的对比来更新相应的属性
            improvement = 0
            self._is_best = False
        else:
            improvement = last_score - self._running_best
            self._running_best = last_score
            self._is_best = True
        return improvement

    #核心逻辑：判断model过拟合程度的相关方法
    def _handle_overfitting(self, last_score, res, std):
        if self._descend_counter == 0:#此前没有处于过拟合状态
            self.info["save_best"] = True#此时应该把今后出现的最优的参数备份下来
            self._over_fit_performance = last_score
        self._descend_counter += min(self.n_tolerance / 3, -res / std)#根据公式记录过拟合程度
        self.over_fitting_flag = 1#将标识“是否处于过拟合状态”的属性设置为1

    #核心逻辑：定义处理正从过拟合状态中恢复的情形的方法
    def _handle_recovering(self, improvement, last_score, res, std):
        #如果model的性能有了飞跃式的进步的话，就认为需要把最优参数备份下来
        if res > 3 * std and self._is_best and improvement > std:
            self.info["save_best"] = True
        new_counter = self._descend_counter - res / std#计算新的过拟合程度
        #如果之前正处于过拟合，而新的过拟合程度不大于0的话，就说明此时model完全从过拟合的状态中恢复了回来
        if self._descend_counter > 0 >= new_counter:#于是就需要把一些属性复原
            self._over_fit_performance = math.inf
            if last_score > self._best_checkpoint_performance:
                self._best_checkpoint_performance = last_score
                if last_score > self._running_best - std:
                    self.info["save_checkpoint"] = True
                    self.info["info"] = ("当前快照 ({}) 似乎工作良好, 保存检查点以备需要恢复时使用".format(len(self._scores)))
            self.over_fitting_flag = 0
        self._descend_counter = max(new_counter, 0)#更新过拟合程度

    #备份最优参数
    def _handle_is_best(self):
        if self._is_best:#如果当前表现是最佳表现的话
            self.info["terminate"] = False#就总是不终止训练
            if self.info["save_best"]:#如果此时需要备份最优参数的话
                self.info["save_checkpoint"] = True
                self.info["save_best"] = False
                self.info["info"] = ("当前快照 ({}) 是目前为止的最佳“检查点”, 进行保存 ".format(len(self._scores)))
                if self.over_fitting_flag:
                    self.info["info"] += "模型过拟合了"
                else:
                    self.info["info"] += "模型表现显著提升"

    #周期性地备份参数
    def _handle_period(self, last_score):
        if len(self._scores) % self.snapshot_ratio == 0 and last_score > self._best_checkpoint_performance:
            self._best_checkpoint_performance = last_score
            self.info["terminate"] = False
            self.info["save_checkpoint"] = True
            self.info["info"] = ("当前快照 ({}) 是目前为止的最佳“检查点”, 以防需要恢复，进行保存".format(len(self._scores)))

    #核心方法，new_metric为model的最新表现
    def check(self, new_metric):
        last_score = new_metric * self.sign#因为乘了sign所以分数总是越大越好
        self._scores.append(last_score)
        n_history = min(self.n_history, len(self._scores))#算出实际的滑动窗口长度
        if n_history == 1:#长度为1无须任何判断，因为模型才刚开始训练
            return self.info
        improvement = self._update_running_info(last_score, n_history)#获取当前模型表现相比于历史最有表现得提升程度
        self.info["save_checkpoint"] = False
        mean = self._running_sum / n_history#滑动均值
        std = math.sqrt(max(self._running_square_sum / n_history - mean ** 2, 1e-12))#求出滑动标准差
        std = min(std, self.std_ceiling)#规定std不能超过std_ceiling这个提高鲁棒性的属性
        if std < self.std_floor:
            if self.flat_flag:
                self._flat_counter += 1#给记录稳定程度的加上1
        else:
            self._flat_counter = max(self._flat_counter - 1, 0)#给记录稳定程度的减1
            res = last_score - mean#算出最新的差res
            if res < -std and last_score < self._over_fit_performance - std:#认为model正陷入过拟合
                self._handle_overfitting(last_score, res, std)
            elif res > std:#认为model正从过拟合状态中恢复
                self._handle_recovering(improvement, last_score, res, std)

        #接下来就是根据上面更新过后的过拟合程度的信息来进行收尾的阶段
        if self._flat_counter >= self.n_tolerance * self.n_history:#如果稳定程度计数超过了阈值
            self.info["info"] = "模型的性能已经没有提升了"#告诉调用者“model的性能已经没有提升了”
            self.info["terminate"] = True#然后把终止训练对应的value设为true
            return self.info#返回所有信息
        if self._descend_counter >= self.n_tolerance:#如果过拟合程度超过了阈值
            self.info["info"] = "模型已经过拟合了"#告诉调用者“model已经过拟合了”
            self.info["terminate"] = True
            return self.info
        self._handle_is_best()#调用相应方法处理是否需要备份最优参数的问题
        self._handle_period(last_score)#调用相应方法，处理周期性地备份参数的问题
        return self.info


class DNDF:
    def __init__(self, n_class, n_tree=10, tree_depth=4):
        self.n_class = n_class
        self.n_tree, self.tree_depth = n_tree, tree_depth
        self.n_leaf = 2 ** (tree_depth + 1)
        self.n_internals = self.n_leaf - 1

    def __call__(self, net, n_batch_placeholder, dtype="output", pruner=None):
        name = "DNDF_{}".format(dtype)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            flat_probabilities = self.build_tree_projection(dtype, net, pruner)
            routes = self.build_routes(flat_probabilities, n_batch_placeholder)
            features = tf.concat(routes, 1, name="Feature_Concat")
            if dtype == "feature":
                return features
            leafs = self.build_leafs()
            leafs_matrix = tf.concat(leafs, 0, name="Prob_Concat")
            return tf.divide(tf.matmul(features, leafs_matrix),float(self.n_tree), name=name)

    def build_tree_projection(self, dtype, net, pruner):
        with tf.name_scope("Tree_Projection"):
            flat_probabilities = []
            fc_shape = net.shape[1].value
            for i in range(self.n_tree):
                with tf.name_scope("Decisions"):
                    p_left = tf.nn.sigmoid(fully_connected_linear(
                        net=net,
                        shape=[fc_shape, self.n_internals],
                        appendix="_tree_mapping{}_{}".format(i, dtype), pruner=pruner
                    ))
                    p_right = 1 - p_left
                    p_all = tf.concat([p_left, p_right], 1)
                    flat_probabilities.append(tf.reshape(p_all, [-1]))
        return flat_probabilities

    def build_routes(self, flat_probabilities, n_batch_placeholder):
        with tf.name_scope("Routes"):
            n_flat_prob = 2 * self.n_internals
            batch_indices = tf.reshape(tf.range(0, n_flat_prob * n_batch_placeholder, n_flat_prob),[-1, 1])
            n_repeat, n_local_internals = int(self.n_leaf * 0.5), 1
            increment_mask = np.repeat([0, self.n_internals], n_repeat)
            routes = [tf.gather(p_flat, batch_indices + increment_mask)for p_flat in flat_probabilities]
            for depth in range(1, self.tree_depth + 1):
                n_repeat = int(n_repeat * 0.5)
                n_local_internals *= 2
                increment_mask = np.repeat(np.arange(n_local_internals - 1, 2 * n_local_internals - 1), 2)
                increment_mask += np.tile([0, self.n_internals], n_local_internals)
                increment_mask = np.repeat(increment_mask, n_repeat)
                for i, p_flat in enumerate(flat_probabilities):
                    routes[i] *= tf.gather(p_flat, batch_indices + increment_mask)
        return routes

    def build_leafs(self):
        with tf.name_scope("Leafs"):
            if self.n_class == 1:#如果是回归问题就定义一些普通的权值矩阵
                local_leafs = [
                    init_w([self.n_leaf, 1], "RegLeaf{}".format(i))
                    for i in range(self.n_tree)
                ]
            else:#否则就定义富有概率意义的权值矩阵
                local_leafs = [
                    tf.nn.softmax(w, name="ClfLeafs{}".format(i))
                    for i, w in enumerate([
                        init_w([self.n_leaf, self.n_class], "RawClfLeafs")
                        for _ in range(self.n_tree)
                    ])
                ]
        return local_leafs

#剪枝的核心实现
class Pruner:
    def __init__(self, alpha=None, beta=None, gamma=None, r=1., eps=1e-12, prune_method="soft_prune"):
        self.alpha, self.beta, self.gamma, self.r, self.eps = alpha, beta, gamma, r, eps
        self.org_ws, self.masks, self.cursor = [], [], -1
        self.method = prune_method
        if prune_method == "soft_prune" or prune_method == "hard_prune":
            if alpha is None:
                self.alpha = 1e-2
            if beta is None:
                self.beta = 1
            if gamma is None:
                self.gamma = 1
            if prune_method == "hard_prune":
                self.alpha *= 0.01
            self.cond_placeholder = None
        elif prune_method == "surgery":
            if alpha is None:
                self.alpha = 1
            if beta is None:
                self.beta = 1
            if gamma is None:
                self.gamma = 0.0001
            self.r = None
            self.cond_placeholder = tf.placeholder(tf.bool, (), name="Prune_flag")
        else:
            raise NotImplementedError("剪枝方法  '{}' 未定义".format(prune_method))

    @property
    def params(self):
        return {"eps": self.eps, "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "max_ratio": self.r, "method": self.method }

    #定义辅助剪枝的方法
    @staticmethod
    def get_w_info(w):
        with tf.name_scope("get_w_info"):
            w_abs = tf.abs(w)
            w_abs_mean, w_abs_var = tf.nn.moments(w_abs, None)
            return w, w_abs, w_abs_mean, tf.sqrt(w_abs_var)

    #定义执行剪枝的方法
    def prune_w(self, w, w_abs, w_abs_mean, w_abs_std):
        self.cursor += 1
        self.org_ws.append(w)
        with tf.name_scope("Prune"):
            if self.cond_placeholder is None:
                log_w = tf.log(tf.maximum(self.eps, w_abs / (w_abs_mean * self.gamma)))
                if self.r > 0:
                    log_w = tf.minimum(self.r, self.beta * log_w)
                self.masks.append(tf.maximum(self.alpha / self.beta * log_w, log_w))
                return w * self.masks[self.cursor]

            self.masks.append(tf.Variable(tf.ones_like(w), trainable=False))

            def prune(i, do_prune):
                def sub():
                    if do_prune:
                        mask = self.masks[i]
                        self.masks[i] = tf.assign(mask, tf.where(
                            tf.logical_and(
                                tf.equal(mask, 1),
                                tf.less_equal(w_abs, 0.9 * tf.maximum(w_abs_mean + self.beta * w_abs_std, self.eps))
                            ),
                            tf.zeros_like(mask), mask
                        ))
                        mask = self.masks[i]
                        self.masks[i] = tf.assign(mask, tf.where(
                            tf.logical_and(
                                tf.equal(mask, 0),
                                tf.greater(w_abs, 1.1 * tf.maximum(w_abs_mean + self.beta * w_abs_std, self.eps))
                            ),
                            tf.ones_like(mask), mask
                        ))
                    return w * self.masks[i]
                return sub

            return tf.cond(self.cond_placeholder, prune(self.cursor, True), prune(self.cursor, False))


#缺失值处理器：只需要关系如何处理连续型特征中的nan
class NanHandler:
    def __init__(self, method, reuse_values=True):
        self._values = None#记录训练集的各个统计量
        self.method = method#缺失值处理的方法，即delete、mean（平均数）、median（中位数）
        self.reuse_values = reuse_values#测试时是否复用训练集的各个统计量

    #缺失值处理的具体方法
    def transform(self, x, numerical_idx, refresh_values=False):
        if self.method is None:
            pass
        elif self.method == "delete":#把存在nan的样本都删除
            x = x[~np.any(np.isnan(x[..., numerical_idx]), axis=1)]
        else:#利用统计量来替换nan
            if self._values is None:
                self._values = [None] * len(numerical_idx)
            for i, (v, numerical) in enumerate(zip(self._values, numerical_idx)):#缺失值处理的主循环
                if not numerical:#只关心连续型特征的缺失值处理，非连续型就跳过
                    continue
                feat = x[..., i]#提取出第i维特征
                mask = np.isnan(feat)#利用np.isnan来获取第i维中nan的位置
                if not np.any(mask):
                    continue
                #如果测试时已有相应的训练集统计量且属性设置为复用统计量且参数设置为不刷新统计量
                if self.reuse_values and not refresh_values and v is not None:
                    new_value = v#把相应的统计量设置为“目标”
                else:#计算相应的统计量，并在把该统计量记录进self._values后，把该统计量设置为“目标”
                    new_value = getattr(np, self.method)(feat[~mask])
                    if self.reuse_values and (v is None or refresh_values):
                        self._values[i] = new_value
                feat[mask] = new_value#用“目标”替换掉第i维特征中nan的位置
        return x

    def reset(self):
        self._values = None

#数据预处理器
class PreProcessor:
    def __init__(self, method, scale_method, eps_floor=1e-4, eps_ceiling=1e12):
        self.method, self.scale_method = method, scale_method
        self.eps_floor, self.eps_ceiling = eps_floor, eps_ceiling
        self.redundant_idx = None
        self.min = self.max = self.mean = self.std = None

    def _scale(self, x, numerical_idx):
        targets = x[..., numerical_idx]
        self.redundant_idx = [False] * len(numerical_idx)
        mean = std = None
        if self.mean is not None:
            mean = self.mean
        if self.std is not None:
            std = self.std
        if self.min is None:
            self.min = targets.min(axis=0)
        if self.max is None:
            self.max = targets.max(axis=0)
        if mean is None:
            mean = targets.mean(axis=0)
        abs_targets = np.abs(targets)
        max_features = abs_targets.max(axis=0)
        if self.scale_method is not None:
            max_features_res = max_features - mean
            mask = max_features_res > self.eps_ceiling
            n_large = np.sum(mask)
            if n_large > 0:
                idx_lst, val_lst = [], []
                mask_cursor = -1
                for i, numerical in enumerate(numerical_idx):
                    if not numerical:
                        continue
                    mask_cursor += 1
                    if not mask[mask_cursor]:
                        continue
                    idx_lst.append(i)
                    val_lst.append(max_features_res[mask_cursor])
                    local_target = targets[..., mask_cursor]
                    local_abs_target = abs_targets[..., mask_cursor]
                    sign_mask = np.ones(len(targets))
                    sign_mask[local_target < 0] *= -1
                    scaled_value = self._scale_abs_features(local_abs_target) * sign_mask
                    targets[..., mask_cursor] = scaled_value
                    if self.mean is None:
                        mean[mask_cursor] = np.mean(scaled_value)
                    max_features[mask_cursor] = np.max(scaled_value)
                warn_msg = "{} 值过大: [{}]{}".format(
                    "{} 这几列包含的".format(n_large) if n_large > 1 else "这一列包含的",
                    ", ".join("{}: {:8.6f}".format(idx, val)
                        for idx, val in zip(idx_lst, val_lst)),
                    ", {} 将通过 '{}' 方法处理".format("它" if n_large == 1 else "他们", self.scale_method)
                )
                print(warn_msg)
                x[..., numerical_idx] = targets
        if std is None:
            if np.any(max_features > self.eps_ceiling):
                targets = targets - mean
            std = np.maximum(self.eps_floor, targets.std(axis=0))
        if self.mean is None and self.std is None:
            self.mean, self.std = mean, std
        return x

    def _scale_abs_features(self, abs_features):
        if self.scale_method == "truncate":
            return np.minimum(abs_features, self.eps_ceiling)
        if self.scale_method == "divide":
            return abs_features / self.max
        if self.scale_method == "log":
            return np.log(abs_features + 1)
        return getattr(np, self.scale_method)(abs_features)

    def _normalize(self, x, numerical_idx):
        x[..., numerical_idx] -= self.mean
        x[..., numerical_idx] /= np.maximum(self.eps_floor, self.std)
        return x

    def _min_max(self, x, numerical_idx):
        x[..., numerical_idx] -= self.min
        x[..., numerical_idx] /= np.maximum(self.eps_floor, self.max - self.min)
        return x

    def transform(self, x, numerical_idx):
        x = self._scale(np.array(x, dtype=np.float32), numerical_idx)
        x = getattr(self, "_" + self.method)(x, numerical_idx)
        return x


__all__ = [
    "init_w", "init_b", "fully_connected_linear", "prepare_tensorboard_verbose",
    "Toolbox", "Metrics", "Losses", "Activations", "TrainMonitor",
    "DNDF", "Pruner", "NanHandler", "PreProcessor"]
