# -*- coding:UTF-8 -*-
import numpy as np
import tensorflow as tf
from init import tokenizer, MAX_LEN

SAVE_PATH = './modelFile/'


def predict(model, token_ids):
    """
    在概率值为前100的词中选取一个词(按概率分布的方式)
    :return: 一个词的编号(不包含[PAD][NONE][START])
    """
    # 预测各个词的概率分布
    # -1 表示只要对最新的词的预测
    # 3: 表示不要前面几个标记符
    _probas = model.predict([token_ids, ])[0, -1, 3:]
    # 按概率降序，取前100
    p_args = _probas.argsort()[-100:][::-1]  # 此时拿到的是索引
    p = _probas[p_args]  # 根据索引找到具体的概率值
    p = p / sum(p)  # 归一
    # 按概率抽取一个
    target_index = np.random.choice(len(p), p=p)
    # 前面预测时删除了前几个标记符，因此编号要补上3位，才是实际在tokenizer词典中的编号
    return p_args[target_index] + 3


# 随机生成一首诗
def generate_random_poem(tokenizer, model, text=""):
    """
    随机生成一首诗
    :param tokenizer: 分词器
    :param model: 古诗模型
    :param text: 古诗的起始字符串，默认为空
    :return: 一首古诗的字符串
    """
    # 将初始字符串转成token_ids，并去掉结束标记[END]
    token_ids = tokenizer.encode(text)[:-1]
    while len(token_ids) < MAX_LEN:
        # 预测词的编号
        target = predict(model, token_ids)
        # 保存结果
        token_ids.append(target)
        # 到达END
        if target == tokenizer.end_id:
            break
    return "".join(tokenizer.decode(token_ids))


# 生成一首藏头诗
def generate_acrostic_poem(tokenizer, model, heads):
    """
    生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 古诗模型
    :param heads: 藏头诗的头
    :return: 一首古诗的字符串
    """
    # token_ids，只包含[START]编号
    token_ids = [tokenizer.start_id, ]
    # 逗号和句号标记编号
    punctuation_ids = {tokenizer.token_to_id("，"), tokenizer.token_to_id("。")}
    content = []
    # 为每一个head生成一句诗
    for head in heads:
        content.append(head)
        # head转为编号id，放入列表，用于预测
        token_ids.append(tokenizer.token_to_id(head))
        # 开始生成一句诗
        target = -1
        while target not in punctuation_ids:  # 遇到逗号、句号，说明本句结束，开始下一句
            # 预测词的编号
            target = predict(model, token_ids)
            # 因为可能预测到END，所以加个判断
            if target > 3:
                # 保存结果到token_ids中，下一次预测还要用
                token_ids.append(target)
                content.append(tokenizer.id_to_token(target))
    return "".join(content)


poetryModel = tf.keras.models.load_model(SAVE_PATH + 'poetryModel')

# print(generate_acrostic_poem(tokenizer, poetryModel, '大'))


def main(heads):
    return generate_acrostic_poem(tokenizer, poetryModel, heads)
