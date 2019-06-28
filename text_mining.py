import json
import jieba
import re
from collections import Counter
from pyecharts.charts import Bar, Pie
import pyecharts.options as opts
from pprint import pprint
import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import wordcloud # 词云展示库
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库

# 格式化文本，去除无关信息
def format_content(content):
    content = content.replace(u'\xa0', u' ')
    content = re.sub(r'\[.*?\]','',content)
    content = re.sub(r'\s*作曲.*\n','',content)
    content = re.sub(r'\s*作词.*\n','',content)
    content = re.sub(r'.*:','',content)
    content = re.sub(r'.*：','',content)
    content = content.replace('\n', ' ')
    return content


# 分词
def word_segmentation(content, stop_words):

    # 使用 jieba 分词对文本进行分词处理
    # jieba.enable_parallel()# windows 不支持
    seg_list = jieba.cut(content, cut_all=False)

    seg_list = list(seg_list)

    # 去除停用词
    word_list = []
    for word in seg_list:
        if word not in stop_words:
            word_list.append(word)

    # 过滤遗漏词、空格
    user_dict = [' ', '哒']
    filter_space = lambda w: w not in user_dict
    word_list = list(filter(filter_space, word_list))

    return word_list

# 词频统计
# 返回前 top_N 个值，如果不指定则返回所有值
def word_frequency(word_list, *top_N):
    if top_N:
        counter = Counter(word_list).most_common(top_N[0])
    else:
        counter = Counter(word_list).most_common()

    return counter


def plot_chart(counter, chart_type='Bar'):

    items = [item[0] for item in counter]
    values = [item[1] for item in counter]

    if chart_type == 'Bar':
        # chart = Bar('词频统计')
        # chart.add('词频', items, values, is_more_utils=True)
        chart = (
            Bar()
            .add_xaxis(items)
            .add_yaxis('词频', values)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title='词频统计'))
            )
    else:
        # chart = Pie('词频统计')
        # chart.add('词频', items, values, is_label_show=True, is_more_utils=True)
        chart = (
            Pie()
            .add_xaxis(items)
            .add_yaxis('词频', values)
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
            .set_global_opts(title_opts=opts.TitleOpts(title='词频统计'))
            )
    
    chart.render()

def showbywordcloud(word_counts):
    mask = np.array(Image.open('./cound/2.jpg'))  # 定义词频背景
    wc = wordcloud.WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf',  # 设置字体格式
        mask=mask,  # 设置背景图
        max_words=200,  # 最多显示词数
        max_font_size=100  # 字体最大值
    )

    wc.generate_from_frequencies(word_counts)  # 从字典生成词云
    image_colors = wordcloud.ImageColorGenerator(mask)  # 从背景图建立颜色方案
    wc.recolor(color_func=image_colors)  # 将词云颜色设置为背景图方案
    plt.imshow(wc)  # 显示词云
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像


def main():
    with open('data/lyric_list.json') as f:
        data = json.load(f)

    # 停用词表来自：
    # https://github.com/XuJin1992/ChineseTextClassifier
    with open('data/stop_words.txt','r', encoding='UTF-8') as f:
        stop_words = f.read().split('\n')
    texts =''
    for text in data:
        texts+=text

    lyric = texts
    lyric = format_content(lyric)

    seg_list = word_segmentation(lyric, stop_words)

    counter = word_frequency(seg_list,100)#词频统计 取前十


    words ={}
    for i in counter:
        words[i[0]] = i[1]
    # pprint(words)
    showbywordcloud(words)
    # plot_chart(counter, 'Pie')
    # plot_chart(counter)
    




if __name__ == '__main__':
    main()