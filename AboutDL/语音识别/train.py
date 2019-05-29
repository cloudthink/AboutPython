import os
import tensorflow as tf
import utils
from tqdm import tqdm
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.python.framework import graph_io

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


# 0.准备训练所需数据------------------------------
data_args = utils.data_hparams()
data_args.data_type = 'train'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = True
data_args.stcmd = True
data_args.batch_size = 50#可以将不一次性训练am和lm，同样显存情况下lm的batch_size可以比am的大许多
train_data = utils.get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = utils.data_hparams()
data_args.data_type = 'dev'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = True
data_args.stcmd = True
data_args.batch_size = 50
dev_data = utils.get_data(data_args)


batch_num = len(train_data.wav_lst) // train_data.batch_size
epochs = 100#两个模型都加入了提前终止判断，可以大一些，反正又到不了

# 1.声学模型训练-----------------------------------
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=False):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        return frozen_graph

from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.pny_vocab)
am_args.gpu_nums = 1
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)

flag = os.path.exists(os.path.join(utils.cur_path,'logs_am/model.h5'))
if flag:
    print('加载声学模型...')
    am.ctc_model.load_weights(os.path.join(utils.cur_path,'logs_am/model.h5'))

if flag and input('已有保存的声学模型，是否继续训练 yes/no:') == 'no':
    pass
else:
    checkpoint = ModelCheckpoint(os.path.join(utils.cur_path,'checkpoint', "model_{epoch:02d}-{val_loss:.2f}.h5"), monitor='val_loss',save_best_only=True)
    eStop = EarlyStopping()#损失函数不再减小后patience轮停止训练
    #tensorboard --logdir=/media/yangjinming/DATA/GitHub/AboutPython/AboutDL/语音识别/logs_am/tbLog/ --host=127.0.0.1
    tensbrd = TensorBoard(log_dir=os.path.join(utils.cur_path,'logs_am/tbLog'))
    batch = train_data.get_am_batch()#获取的是生成器
    dev_batch = dev_data.get_am_batch()
    validate_step = 200#取N个验证的平均结果

    history = am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[checkpoint,eStop,tensbrd],
        workers=1, use_multiprocessing=False,verbose=1,validation_data=dev_batch, validation_steps=validate_step)

    am.ctc_model.save_weights(os.path.join(utils.cur_path,'logs_am/model.h5'))
    #写入序列化的 PB 文件
    with keras.backend.get_session() as sess:
        frozen_graph = freeze_session(sess, output_names=['the_inputs','dense_2/truediv'])
        graph_io.write_graph(frozen_graph, os.path.join(utils.cur_path,'logs_am'),'amModel.pb', as_text=False)


# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
import numpy as np
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.2
lm_args.lr = 0.0003
lm_args.is_training = True
lm = Lm(lm_args)

batch_num_list = [i for i in range(batch_num)]#为进度条显示每一个epoch中的进度用
loss_list=[]#记录每一步平均损失的列表，实现提前终止训练功能：每次取出后N个数据的平均值和当前的平均损失值作比较
with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists(os.path.join(utils.cur_path,'logs_lm/checkpoint')):
        print('加载语言模型中...')
        latest = tf.train.latest_checkpoint(os.path.join(utils.cur_path,'logs_lm'))
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    #tensorboard --logdir=/media/yangjinming/DATA/GitHub/AboutPython/AboutDL/语音识别/logs_lm/tensorboard --host=127.0.0.1
    writer = tf.summary.FileWriter(os.path.join(utils.cur_path,'logs_lm/tensorboard'), tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in tqdm(batch_num_list,ncols=90):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        avg_loss = total_loss/batch_num
        print('步数', k+1, ': 平均损失值 = ', avg_loss)

        loss_list.append(avg_loss)
        if len(loss_list)>1 and avg_loss>np.mean(loss_list[-5:])-0.0015:#平均每个epoch下降不到0.0005则终止
            #if input('模型性能已无法提升，是否提前结束训练？ yes/no:')=='yes':
                epochs = k+1#为后面保存模型时记录名字用
                break
        
    saver.save(sess, os.path.join(utils.cur_path,'logs_lm/model_%d' % (epochs + add_num)))
    writer.close()
    # 写入序列化的 PB 文件
    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['x','y','preds'])
    with tf.gfile.GFile(os.path.join(utils.cur_path,'logs_lm','lmModel.pb'), mode='wb') as f:
        f.write(constant_graph.SerializeToString())