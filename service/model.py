# -*- coding:UTF-8 -*-
import math
import numpy as np
import tensorflow as tf
from init import tokenizer, poetrys, titles


SAVE_PATH = './modelFile/'


class DataSet:
    """
    古诗数据集生成器
    """

    def __init__(self, data, tokenizer, batch_size):
        # 数据集
        self.data = data
        self.total_size = len(self.data)
        # 分词器，用于词转编号
        self.tokenizer = tokenizer
        # 每批数据量
        self.batch_size = BATCH_SIZE
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))

    def pad_line(self, line, length, padding=None):
        """
        对齐单行数据
        """
        if padding is None:
            padding = self.tokenizer.pad_id

        padding_length = length - len(line)

        if padding_length > 0:
            return line + [padding] * padding_length
        else:
            return line[:length]

    def __len__(self):
        return self.steps

    def __iter__(self):
        # 打乱数据
        np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for start in range(0, self.total_size, self.batch_size):

            end = min(start + self.batch_size, self.total_size)
            data = self.data[start:end]

            max_length = max(map(len, data))

            batch_data = []
            for str_line in data:
                # 对每一行诗词进行编码、并补齐padding
                encode_line = self.tokenizer.encode(str_line)
                # 加2是因为tokenizer.encode会添加START和END
                pad_encode_line = self.pad_line(encode_line, max_length + 2)
                batch_data.append(pad_encode_line)
            batch_data = np.array(batch_data)
            # yield 特征、标签
            yield batch_data[:, :-1], batch_data[:, 1:]

    def generator(self):
        while True:
            yield from self.__iter__()


BATCH_SIZE = 32

poetryDataset = DataSet(poetrys, tokenizer, BATCH_SIZE)

poetryModel = tf.keras.Sequential([
    # 词嵌入层
    tf.keras.layers.Embedding(input_dim=tokenizer.dict_size, output_dim=100),
    # 第一个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    # 第二个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    # 利用TimeDistributed对每个时间步的输出都做Dense操作(softmax激活)
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
        tokenizer.dict_size, activation='softmax')),
])
poetryModel.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy
)

poetryModel.fit(
    poetryDataset.generator(),
    steps_per_epoch=poetryDataset.steps,
    epochs=10
)
poetryModel.save(SAVE_PATH + 'poetryModel', save_format='HDF5')
