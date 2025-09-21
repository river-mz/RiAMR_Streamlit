import pandas as pd
import numpy as np
import pickle
import os
from scipy.stats import norm
import streamlit as st
import altair as alt
import pandas as pd



def predict_with_ci(df, prescription_columns, save_dir = '/ibex/user/xiex/ide/AMR_proj2/model_training/training/lr_xgb/training_test_april_8_balanced_5_fold/clean_model_5_fold'):
    print(df.shape)

    # 计算95%的预测CI
    def calculate_prediction_intervals(predictions, confidence=0.95):
        print('Hi', predictions.shape)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions, ddof=1)
        n = len(predictions)
        
        # 计算Z值
        z = norm.ppf((1 + confidence) / 2)
        
        # 计算CI
        ci_lower = mean_pred - z * (std_pred / np.sqrt(n))
        ci_upper = mean_pred + z * (std_pred / np.sqrt(n))
        
        return ci_lower, ci_upper

    # 存储结果
    results_dict = {}

    for prescription in prescription_columns:
        print(f"Loading models for {prescription}...")
        
        predictions = []
        model_num = 5
        # 加载每个模型并进行预测
        for i in range(model_num):
            model_path = os.path.join(save_dir, f"xgb_model_{prescription}_fold_{i}.pkl")
            print('model ', i)
            print(model_path)
            if os.path.exists(model_path):
                print('hi')
                with open(model_path, 'rb') as model_file:
                    xgb_model = pickle.load(model_file)
                print('model predictted')
                # 预测
                y_proba_xgb = xgb_model.predict_proba(df)[:, 1]
                predictions.append(y_proba_xgb)
        
        predictions = np.array(predictions)
        print(predictions.shape)
        
        # 计算每个样本的均值和CI
        mean_predictions = np.mean(predictions, axis=0)
        ci_bounds = [calculate_prediction_intervals(predictions[:, j]) for j in range(predictions.shape[1])]
        
        # 存储结果
        results_dict[prescription] = {
            'mean_predictions': mean_predictions,
            'confidence_intervals': ci_bounds
        }

    # 创建结果 DataFrame
    result_df = pd.DataFrame(index=df.index)

    for prescription in prescription_columns:
        mean_preds = results_dict[prescription]['mean_predictions']
        ci_bounds = results_dict[prescription]['confidence_intervals']
        
        result_df[f'{prescription}_prediction'] = mean_preds
        result_df[f'{prescription}_ci_lower'] = [ci[0] for ci in ci_bounds]
        result_df[f'{prescription}_ci_upper'] = [ci[1] for ci in ci_bounds]

    return result_df





# visualization




def plot_antibiotic_probabilities(data_path, antibiotics):
    # 读取数据
    data = pd.read_csv(data_path, header=0)

    # 定义抗生素的列名和颜色
    # antibiotics = ['resistance_nitrofurantoin', 'resistance_sulfamethoxazole', 'resistance_ciprofloxacin', 'resistance_levofloxacin']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors = ['#1f77b4', '#ff7f0e',]

    # 初始化空的图表列表
    charts = []

    # 遍历每个数据组
    for i, row in data.iterrows():
        # 将当前组数据构建成DataFrame
        df = pd.DataFrame({
            'Antibiotic': antibiotics,
            'Probability': [row[f'{antibiotic}_prediction'] for antibiotic in antibiotics],
            'CI_lower': [row[f'{antibiotic}_ci_lower'] for antibiotic in antibiotics],
            'CI_upper': [row[f'{antibiotic}_ci_upper'] for antibiotic in antibiotics]
        })

        # 创建条形图
        bar_chart = alt.Chart(df).mark_bar(size=20).encode(
            x=alt.X('Probability:Q', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('Antibiotic:N', axis=alt.Axis(title='', labelFontSize=12)),
            color=alt.Color('Antibiotic:N', scale=alt.Scale(range=colors), legend=None),
            tooltip=[alt.Tooltip('Probability', format='.1%'), alt.Tooltip('CI_lower', title='CI Lower', format='.2f'), alt.Tooltip('CI_upper', title='CI Upper', format='.2f')]
        )

        # 置信区间的线条
        # ci_rule = alt.Chart(df).mark_rule(color='gray', strokeWidth=1.5).encode(
        #     x='CI_lower:Q',
        #     x2='CI_upper:Q',
        #     y=alt.Y('Antibiotic:N')
        # )

        ci_rule = alt.Chart(df).mark_rule(color='gray', strokeWidth=1.5).encode(
            x=alt.X('CI_lower:Q', title='Probability, Lower CI'),
            x2=alt.X2('CI_upper:Q', title='Upper CI (95%)'),
            y=alt.Y('Antibiotic:N')
        )


        # CI的终端点
        ci_endpoints = alt.Chart(df).mark_tick(color='gray', thickness=2, size=10).encode(
            x='CI_lower:Q',
            y='Antibiotic:N'
        ) + alt.Chart(df).mark_tick(color='gray', thickness=2, size=10).encode(
            x='CI_upper:Q',
            y='Antibiotic:N'
        )

        # 添加文本标签显示百分比
        text = bar_chart.mark_text(
            align='left',
            baseline='middle',
            dx=5,  # 文本偏移
            color='white'
        ).encode(
            text=alt.Text('Probability:Q', format='.1%')
        )

        # 合并条形图、置信区间线条和终端点，并增加标题表示组编号
        chart = (bar_chart + ci_rule + ci_endpoints + text).properties(
            title=f'sample {i+1}',
            width=300,
            height=150
        )

        # 将图表添加到列表中
        charts.append(chart)

    # 将所有图表按列组合，并显示
    with st.container():
        st.altair_chart(alt.vconcat(*charts), use_container_width=True)

