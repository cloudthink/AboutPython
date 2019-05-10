#-*- coding: UTF-8 -*-
import os
import sys
root_path = os.path.abspath("../../")
if root_path not in sys.path:
    sys.path.append(root_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import copy

from DistNN.NN import DistAdvanced

base_params = {
    "data_info": {
        "data_folder": "_Data",
        "file_type": "txt"
    },
    "model_param_settings": {"metric": "auc", "n_epoch": 32,"max_epoch": 1024, 
                             "activations": ["relu", "relu", "relu", "relu"]},
    "pre_process_settings": {}
}

grid_params = {
    "model_param_settings": {
        "lr": [1e-1,1e-2, 1e-3],
        'learning_rate_decay':[None,0.999],#用学习率衰减时初始学习率可以比较大，所以lr中加入0.1（甚至可以再大点）
        'lrMethod':[None,'natural_exp','inverse_time'],
        "loss": ["mse", "cross_entropy"],
        'regularization_Rate':[0.001,0.003,None,0.03,0.1,0.3,1,3,10]#正则率，只有在使用正则化优化技术时才会获取这个参数的值，使用时不赋值默认0.01(当前列表采用Playground参数列表)
    },
    "model_structure_settings": {
        "use_pruner": [False, True],
        "use_regularization":['l1','l2',None],#正则化优化技术
        "use_wide_network": [False, True],
        "use_dndf_pruner":[False, True],
        "hidden_units": [[128, 128, 128, 128], [256, 256,256,256], [512, 512,512,512]],
        "use_batch_norm":[False, True],
        "batch_size":[64,128],
    },
}#目前这几个参数组合去掉一些明显无意义的还有21024个组合……所以轻易不要网格搜索

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #numerical_idx = [False,False,False,False]
    #base_params["model_structure_settings"]["use_dndf_pruner"] = False
    #base_params["model_structure_settings"]["use_pruner"] = True
    #base_params["model_structure_settings"]["use_wide_network"] = False
    #base_params["pre_process_settings"]["reuse_mean_and_std"] = True

    numerical_idx = np.load("_Data/idx/38.npy")#获取numerical_idx
    base_params["model_param_settings"]["sess_config"] = config
    base_params["data_info"]["numerical_idx"] = numerical_idx

    model_name = '38'
    #模型创建，参数搜索（搜索完成后设置成最佳参数设置），用最佳参数设置再拟合一次，拟合结束后会回滚到最佳存档点，然后进行保存
    dist = DistAdvanced(model_name, **base_params).grid_search(grid_params, "dict_first",single_search_time_limit=360,param_search_time_limit=7200,k=5)
    #dist = DistAdvanced(model_name, **base_params)#.k_random(cv_rate=0.1, test_rate=0.1)
    dist.fit()
    dist = DistAdvanced("38", **base_params).grid_search(grid_params, "dict_first")
    #dist.save('Best'+model_name,'D:\\baseFramework\\_Data\\')

def loadModel():#加载模型，直接预测    
    dist = DistAdvanced("1120")#创建模型
    dist.load(run_id='Best1120',path='D:\\baseFramework\\_Data\\')#加载训练时保存的最佳模型
    x = '28.7967 16.0021 2.6449 0.3918 0.1982 27.7004 22.011 -8.2027 40.092 81.8828'
    y = '187.1814 53.0014 3.2093 0.2876 0.1539 -167.3125 -168.4558 31.4755 52.731 272.3174'
    x = [elem if elem != 'nan' else 0.0 for elem in x.strip().split(' ')]
    y = [elem if elem != 'nan' else 0.0 for elem in y.strip().split(' ')]
    x = [np.array(x),np.array(y)]
    print(dist.predict_labels(x))

if __name__ == '__main__':
    main()
    #loadModel()