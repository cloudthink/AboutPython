import os
import tensorflow as tf
import utils
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config))


# 0.准备训练所需数据------------------------------
data_args = utils.data_hparams()
data_args.data_type = 'train'
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = True
data_args.stcmd = False
data_args.batch_size = 2
data_args.data_length = None
data_args.shuffle = True
train_data = utils.get_data(data_args)


# 0.准备验证所需数据------------------------------
data_args = utils.data_hparams()
data_args.data_type = 'dev'
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = True
data_args.stcmd = False
data_args.batch_size = 2
data_args.data_length = None
data_args.shuffle = True
dev_data = utils.get_data(data_args)


# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.pny_vocab)
am_args.gpu_nums = 1
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)


if os.path.exists(os.path.join(utils.cur_path,'logs_am/model.h5')):
    print('加载声学模型...')
    am.ctc_model.load_weights(os.path.join(utils.cur_path,'logs_am/model.h5'))

epochs = 100
batch_num = len(train_data.wav_lst) // train_data.batch_size

# checkpoint
checkpoint = ModelCheckpoint(os.path.join(utils.cur_path,'checkpoint', "model_{epoch:02d}-{val_loss:.2f}.h5"), monitor='val_loss',save_best_only=True)
eStop = EarlyStopping(patience=2)#损失函数不再减小后patience轮停止训练
#tensbrd = TensorBoard(log_dir='./tmp/tbLog')
cbList =[checkpoint,eStop]
batch = train_data.get_am_batch()#获取的是生成器
dev_batch = dev_data.get_am_batch()
validate_step = 100#取N个验证的平均结果
history = am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=cbList,
    workers=1, use_multiprocessing=False,verbose=1,validation_data=dev_batch, validation_steps=validate_step)
am.ctc_model.save_weights(os.path.join(utils.cur_path,'logs_am', 'model.h5'))


# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
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

epochs = 100
with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists(os.path.join(utils.cur_path,'logs_lm/checkpoint')):
        print('加载语言模型中...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter(os.path.join(utils.cur_path,'logs_lm/tensorboard'), tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        print('步数', k+1, ': 平均损失值 = ', total_loss/batch_num)
    saver.save(sess, os.path.join(utils.cur_path,'logs_lm/model_%d' % (epochs + add_num)))
    writer.close()