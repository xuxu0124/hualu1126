#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: XuXu

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio

pio.templates.default = 'plotly_white'


# 01 设定变量
# ------------------------------------------------------------------------------------------------------
path_result_sim = './results/result_sim.xlsx'
str_version_date = '2023-11-26'


# 02 读取数据
# ------------------------------------------------------------------------------------------------------
df_original = pd.read_excel(path_result_sim, '01-df_original')
df_number = pd.read_excel(path_result_sim, '02-df_number')
df_train = pd.read_excel(path_result_sim, '03-df_train')
df_test = pd.read_excel(path_result_sim, '04-df_test')
df_vs = pd.read_excel(path_result_sim, '05-df_vs')
df_real = pd.read_excel(path_result_sim, '06-df_real')
df_mean_train = pd.read_excel(path_result_sim, '07-df_mean_train')
df_pred_nn = pd.read_excel(path_result_sim, '08-df_pred_nn')
df_pred_xgb = pd.read_excel(path_result_sim, '09-df_pred_xgb')
df_pred_2p = pd.read_excel(path_result_sim, '10-df_pred_2p')
df_input_output = pd.read_excel(path_result_sim, '11-df_input_output')


# 03 生成相关热图
# ------------------------------------------------------------------------------------------------------
# fig_train_heatmap = px.imshow(
#     df_corr, color_continuous_scale='RdYlBu_r', text_auto=True, color_continuous_midpoint=0, aspect='auto',
#     title='定制化相关热图  兼容连续型特征、有序分类特征和无序分类特征'
# )


# 04 显示
# ------------------------------------------------------------------------------------------------------
st.markdown(f'## 仿真试验结果-{str_version_date}')
st.markdown('---')

st.markdown('### 一、材料')
st.markdown('''
&emsp;&emsp;本次仿真试验数据来自于上次训练树模型的 **测试数据**，具体的描述请查阅其来源。
''')

st.markdown('### 二、方法')
st.markdown('''
&emsp;&emsp;仿真试验主要包括3个步骤，分别是 **数据读取**、 **数据预处理** 以及 **数据建模**。
''')

st.markdown('#### （一）、数据读取')
st.markdown('''
&emsp;&emsp;1、读取原始记录表与哑变量化表格；\n
&emsp;&emsp;2、对数据进行列筛选，得到由模型的输入与输出变量组成的表格；\n
&emsp;&emsp;3、对表格进行 **01归一化**。
''')

st.markdown('#### （二）、数据预处理')
st.markdown('''
&emsp;&emsp;1、使用”2022年PCI（缝补）”列进行 **采样误差** 预估；\n
&emsp;&emsp;2、使用 **双参模型** 进行逐年 **指标生成**；\n
&emsp;&emsp;3、使用 **等距抽样** 拆分 **训练集** 和 **测试集**。
''')

st.markdown('#### （三）、数据建模')
st.markdown('''
&emsp;&emsp;1、使用 **训练集** 训练神经网络模型；\n
&emsp;&emsp;2、使用 **训练集** 训练XGBoost模型；\n
&emsp;&emsp;3、使用 **测试集** 计算双参模型R方，用于充当 **背景噪声**，提高批间稳定性；\n
&emsp;&emsp;4、使用 **测试集** 计算神经网络和XGBoost的R方，用于备选建模 **方法之间的对比**。\n
''')


st.markdown('### 三、保存结果')
st.markdown('''
&emsp;&emsp;保存结果包括11张表格，对应excel文件中的11张sheets。
11张结果表格分别是《01-df_original》《02-df_number》《03-df_train》《04-df_test》《05-df_vs》
《06-df_real》《07-df_mean_train》《08-df_pred_nn》《09-df_pred_xgb》《10-df_pred_2p》《11-df_input_output》。
这11张结果表格的说明如下表所示。
''')
list_info = [
    ['01-df_original', '原始数据表，方便查阅每一段公路的基本信息。'],
    ['02-df_number', '哑变量化的表格，经过了数值化后的结果。'],
    ['03-df_train', '用于训练模型的训练数据集。'],
    ['04-df_test', '用于测试模型的测试数据集。'],
    ['05-df_vs', '三种方法的R方对比，其中R方取值从0到1。'],
    ['06-df_real', '真实值矩阵，每一行是测试路段，每一列是预测年份。'],
    ['07-df_mean_train', '真实值矩阵的各年平均。'],
    ['08-df_pred_nn', '神经网络模型的预测矩阵。'],
    ['09-df_pred_xgb', 'XGBoost模型的预测矩阵。'],
    ['10-df_pred_2p', '双参模型的预测矩阵，其中混入了噪声。'],
    ['11-df_input_output', '模型的输入特征和输出变量信息。']
]
list_info = list(zip(*list_info))
df_res_tab_info = pd.DataFrame({
    '表名': list_info[0],
    '说明': list_info[1]
})
with st.expander('🧾 结果表格说明'):
    st.dataframe(df_res_tab_info, use_container_width=True)

st.markdown('''
&emsp;&emsp;全部11张表格如下所示。
其中 **建模前** 的数据准备请查看《数据表格: 01-04》，
**建模后** 的结果对比请查看《结果表格: 05-10》，
**特征信息** 请查看《特征信息: 11》。
''')
with st.expander('🧾 数据表格: 01-04'):
    list_tabs = st.tabs([
        '01-df_original',
        '02-df_number',
        '03-df_train',
        '04-df_test'
    ])
    with list_tabs[0]:
        st.dataframe(df_original, use_container_width=True)
    with list_tabs[1]:
        st.dataframe(df_number, use_container_width=True)
    with list_tabs[2]:
        st.dataframe(df_train, use_container_width=True)
    with list_tabs[3]:
        st.dataframe(df_test, use_container_width=True)

with st.expander('🧾 结果表格: 05-10'):
    list_tabs = st.tabs([
        '05-df_vs',
        '06-df_real',
        '07-df_mean_train',
        '08-df_pred_nn',
        '09-df_pred_xgb',
        '10-df_pred_2p'
    ])
    with list_tabs[0]:
        st.dataframe(df_vs)
    with list_tabs[1]:
        st.dataframe(df_real, use_container_width=True)
    with list_tabs[2]:
        st.dataframe(df_mean_train, use_container_width=True)
    with list_tabs[3]:
        st.dataframe(df_pred_nn, use_container_width=True)
    with list_tabs[4]:
        st.dataframe(df_pred_xgb, use_container_width=True)
    with list_tabs[5]:
        st.dataframe(df_pred_2p, use_container_width=True)

with st.expander('🧾 特征信息: 11'):
    st.dataframe(df_input_output, use_container_width=True)


st.markdown('### 四、重要结果可视化')
st.markdown('''
&emsp;&emsp;对三种方法的R方对比、三种方法的预测矩阵对比以及去除噪声后的两种方法误差对比进行可视化。
可视化结果如下所示。
''')

df_vs_2 = df_vs.copy()
df_vs_2['method'] = df_vs_2.method.map({'nn': '神经网络', 'xgb': 'XGBoost', '2p': '双参模型'})
st.write(df_vs_2)
df_vs_2.columns = ['建模方法', 'R方']
fig_bar_r2 = px.bar(
    df_vs_2, x='建模方法', y='R方', color='建模方法', range_y=[0, 1],
    title='神经网络、XGBoost和双参模型对测试集预测的R方对比'
)
with st.expander('📊 01 - 三种方法的R方对比'):
    st.plotly_chart(fig_bar_r2, use_container_width=True)


# 3个预测矩阵热图
# ----------------------------------------------------------------------
zmin, zmax = 75, 100
fig_heatmap_nn = px.imshow(
    df_pred_nn, color_continuous_scale='RdBu_r', aspect='auto',
    zmin=zmin, zmax=zmax, title='神经网络预测矩阵'
)
fig_heatmap_nn.update_xaxes(showticklabels=False)
fig_heatmap_nn.update_yaxes(showticklabels=False)

fig_heatmap_xgb = px.imshow(
    df_pred_xgb, color_continuous_scale='RdBu_r', aspect='auto',
    zmin=zmin, zmax=zmax, title='XGBoost预测矩阵'
)
fig_heatmap_xgb.update_xaxes(showticklabels=False)
fig_heatmap_xgb.update_yaxes(showticklabels=False)

fig_heatmap_2p = px.imshow(
    df_pred_2p, color_continuous_scale='RdBu_r', aspect='auto',
    zmin=zmin, zmax=zmax, title='双参模型预测矩阵'
)
fig_heatmap_2p.update_xaxes(showticklabels=False)
fig_heatmap_2p.update_yaxes(showticklabels=False)

with st.expander('📊 02 - 三种方法的预测结果矩阵对比'):
    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        col_1.plotly_chart(fig_heatmap_nn, use_container_width=True)
    with col_2:
        col_2.plotly_chart(fig_heatmap_xgb, use_container_width=True)
    with col_3:
        col_3.plotly_chart(fig_heatmap_2p, use_container_width=True)


# 误差矩阵热图
# ----------------------------------------------------------------------
zmin, zmax = -4, 4
fig_error_nn = px.imshow(
    df_pred_2p - df_pred_nn, color_continuous_scale='RdBu_r', aspect='auto',
    zmin=zmin, zmax=zmax, title='神经网络误差矩阵'
)
fig_error_nn.update_xaxes(showticklabels=False)
fig_error_nn.update_yaxes(showticklabels=False)

fig_error_xgb = px.imshow(
    df_pred_2p - df_pred_xgb, color_continuous_scale='RdBu_r', aspect='auto',
    zmin=zmin, zmax=zmax, title='XGBoost误差矩阵'
)
fig_error_xgb.update_xaxes(showticklabels=False)
fig_error_xgb.update_yaxes(showticklabels=False)

with st.expander('📊 03 - 两种方法预测误差的对比'):
    col_1, col_2 = st.columns(2)
    with col_1:
        col_1.plotly_chart(fig_error_nn, use_container_width=True)
    with col_2:
        col_2.plotly_chart(fig_error_xgb, use_container_width=True)

