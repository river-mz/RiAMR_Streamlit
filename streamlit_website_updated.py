import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import torch
import pickle
import os
import torch.nn as nn
import json
from utils_updated import preprocess_data
import shap 
import matplotlib.pyplot as plt
from predict_plot import predict_with_ci, plot_antibiotic_probabilities
import base64

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import torch
import pickle
import os
import torch.nn as nn
import json
from utils_updated import preprocess_data
import shap 
import matplotlib.pyplot as plt
from predict_plot import predict_with_ci, plot_antibiotic_probabilities
import base64

# 准备 GIF 文件路径
gif_path = "/ibex/user/xiex/ide/AMR_proj2/streamlit_running/resource/animated_icon.gif"


# 将 GIF 文件读取为二进制并进行 Base64 编码

@st.cache_data
def load_gif(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

data_url = load_gif(gif_path)

# # 在页面顶部插入 GIF 和标题，确保它们在同一行
st.markdown(
    f'''
    <div style="display: flex; align-items: center;">
        <img src="data:image/gif;base64,{data_url}" alt="Animated Icon" style="width:60px; height:60px; margin-right: 10px;"/>
        <h1>RiAMR</h1>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    f'''
    <div>
        <h2>Robust and Interpretable Antimicrobial Resistance Prediction from Electronic Health Records</h2>
    </div>
    ''',
    unsafe_allow_html=True
)

# 设置页面标题
st.markdown(
    """
    <style>
    .block {
        background-color: #f8f9fa;  /* Light grey background */
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;  /* Space between blocks */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .streamlit-container {
        max-width: 1000px;  /* Set the maximum width of the container */
        margin: auto;  /* Center the container */
    }

        /* Custom style for the selectbox to fit the container */
    .custom-selectbox .stSelectbox {
        width: 100%;  /* Make selectbox width 100% of its container */
    }
    </style>
    """,
    unsafe_allow_html=True

)

# Antibiotic options and corresponding models
antibiotics_options = {

    "ciprofloxacin": {
        "lr_model": "lr_model_resistance_ciprofloxacin.pth",
        "xgb_model": "xgb_model_resistance_ciprofloxacin.pkl"
    },
    "nitrofurantoin": {
        "lr_model": "lr_model_resistance_nitrofurantoin.pth",
        "xgb_model": "xgb_model_resistance_nitrofurantoin.pkl"
    }
}


# Define model directory
model_dir = '/ibex/user/xiex/ide/AMR_proj2/model_training/training/lr_xgb/training_test_april_8_balanced_5_fold/selected_model/'
# xgb_model_resistance_ciprofloxacin.pkl

# Load the feature columns from the JSON file
feature_columns_path = "/ibex/user/xiex/ide/AMR_proj2/streamlit_running/feature_columns_updated.json"
with open(feature_columns_path, "r") as file:
    feature_columns = json.load(file)

results_df = pd.DataFrame()


def convert_gender_to_bool(value):
    if isinstance(value, str):  # 检查是否为字符串
        if value.lower() == "Male":  # 将 "true" 转换为 True
            return 1
        elif value.lower() == "Female":  # 将 "false" 转换为 False
            return 0
    return np.nan  # 对于其他情况，返回布尔值


def convert_dept_to_num(value):
    if isinstance(value, str):  # 检查是否为字符串
        if value.lower() == "ER":  # 将 "true" 转换为 True
            return 0
        elif value.lower() == "ICU":  # 将 "false" 转换为 False
            return 1
        elif value.lower() == "IP":  # 将 "false" 转换为 False
            return 2
        elif value.lower() == "OP":  # 将 "false" 转换为 False
            return 3
    return np.nan  # 对于其他情况，返回布尔值

def convert_organism_to_num(value):
    if isinstance(value, str):  # 检查是否为字符串
        if value.lower() == "acinetobacter baumannii":  # 将 "true" 转换为 True
            return 0
        elif value.lower() == "enterobacter":  # 将 "true" 转换为 True
            return 1
        elif value.lower() == "enterococcus faecium":  # 将 "false" 转换为 False
            return 2
        elif value.lower() == "klebsiella pneumoniae":  # 将 "false" 转换为 False
            return 3
        elif value.lower() == "pseudomonas aeruginosa":  # 将 "false" 转换为 False
            return 4
        elif value.lower() == "staphylococcus aureus":  # 将 "false" 转换为 False
            return 5
    return np.nan  # 对于其他情况，返回布尔值

def convert_to_bool(value):
    if isinstance(value, str):  # 检查是否为字符串
        if value.lower() == "true":  # 将 "true" 转换为 True
            return 1
        elif value.lower() == "false":  # 将 "false" 转换为 False
            return 0
    return np.nan  # 对于其他情况，返回布尔值


@st.cache_resource  # Cache the model loading function
def load_xgb_model(antibiotic):
    model_path = os.path.join(model_dir, antibiotics_options[antibiotic]["xgb_model"])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Function to display demo DataFrame
def show_demo_data():
    demo_df = pd.read_csv('./test_demo_input3.csv')
    st.write("Demo Data & Data Type:")
    st.dataframe(demo_df.reset_index(drop=True), use_container_width=True,  hide_index=True)


# 创建选项菜单
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Predict", "About"],
        icons=['house', 'bar-chart', 'info-circle'],
        menu_icon="cast", default_index=0)

# 根据用户选择的菜单显示不同的内容
if selected == "Home":
    # Title for the main application
    # st.title("RiAMR -- Robust and Interpretable Antimicrobial Resistance Prediction from Electronic Health Records")
    st.markdown(
        """
        <h4>Welcome to the AMR Prediction Application!</h4>
        <p>This application is designed to assist healthcare professionals in predicting antimicrobial resistance (AMR) based on patient data. 
        By leveraging advanced machine learning models, it provides insights into the likelihood of resistance to various antibiotics, enabling better-informed treatment decisions.
        </p>
        """,
        unsafe_allow_html=True
    )


    overview_image_path = f'/ibex/user/xiex/ide/AMR_proj2/streamlit_running/view2.jpg'

    # overview_image_path = f'/ibex/user/xiex/ide/AMR_proj2/streamlit_running/overview3.jpg'

    st.image(overview_image_path, caption=f'RiAMR Framework')


    # feature
    with st.container():
        st.markdown(
            """
            <div class="block">
            <h3>✨ Key Features</h3>
            <ul class="features">
                <li><strong>🖥️ User-Friendly Interface:</strong> Easily upload your patient data in CSV format or input it manually.</li>
                <li><strong>⚡ Model Selection:</strong> Choose from multiple antibiotics for resistance prediction, including nitrofurantoin, sulfamethoxazole, ciprofloxacin, and levofloxacin.</li>
                <li><strong>📊 SHAP Visualization:</strong> Understand the model's predictions through SHAP (SHapley Additive exPlanations) values, which highlight the contribution of each feature to the prediction.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Adjust the stylable_container to properly contain all elements
    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            width: 100%;  /* Ensure full width */
            max-width: 800px;  /* Set a max width to prevent overflow */
            margin: 0 auto;  /* Center the container */
            }
            """,
    ):
        st.markdown(
            """
            <h3>🤖 How the Model Makes Predictions</h3>
            <p>The AMR Prediction Application utilizes interpretable machine learning model, including Logistic Regression and XGBoost, to predict antimicrobial resistance based on patient data. 
            The model's predictions are complemented by SHAP (SHapley Additive exPlanations) values, which provide <br>
            insights into how each feature contributes to the final prediction.</p>
            
            """,
            unsafe_allow_html=True
        )

        with st.container():
            selected_antibiotic_global = st.selectbox("Select an antibiotic for global SHAP plot:", list(antibiotics_options.keys()), key="selectbox_global")

        # Display the global SHAP plot based on the selected antibiotic
        if selected_antibiotic_global:
            global_shap_image_path = f'/ibex/project/c2205/AMR_dataset_peijun/model_prediction/shap_Ours_Oct_19/shap_xgb_resistance_{selected_antibiotic_global}.png'
            st.image(global_shap_image_path, caption=f'Global SHAP value in XGBoost for {selected_antibiotic_global}', width=620)

elif selected == "Predict":
    # st.title("RiAMR -- Robust and Interpretable Antimicrobial Resistance Prediction from Electronic Health Records")


    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            width: 100%;  /* Ensure full width */
            max-width: 800px;  /* Set a max width to prevent overflow */
            margin: 0 auto;  /* Center the container */
            }
            """,
        ):

        st.markdown(
            """
            <h3>Input Data</h3>
            """,
            unsafe_allow_html=True
        )
        input_method = st.radio("Choose input method:", ("Upload CSV", "Manual Input"))
        final_df = pd.DataFrame()

        
        if input_method == "Upload CSV":
            # # Upload file section
            # Example CSV Download
            # Show example file for users
            with st.container():
                st.markdown("📄 Example Input File (example input file to help you format your CSV correctly):")
                try:
                    example_df = pd.read_csv('/ibex/user/xiex/ide/AMR_proj2/streamlit_running/test11_before_preprocessing.csv')

                    st.data_editor(
                        example_df.head(),  # Show only the first few rows
                        disabled=True,
                        use_container_width=True,
                        column_config={
                            'age': st.column_config.NumberColumn('Age', help="Enter the patient’s age in years."),
                            'race': st.column_config.SelectboxColumn('Race (White/Non-White)', help="Whether patient is White."),
                            'veteran': st.column_config.SelectboxColumn('Veteran', help="Is the patient a veteran?"),
                            'gender': st.column_config.SelectboxColumn('Gender', help="Male or Female"),
                            'BMI': st.column_config.NumberColumn('BMI', help="Body Mass Index (e.g., 24.5)"),
                            'LOS': st.column_config.NumberColumn('Length of Stay', help="Days stayed in hospital"),
                            'organism_name_before': st.column_config.SelectboxColumn('Organism', help="Target organism (optional)"),
                            'department_type_before': st.column_config.SelectboxColumn('Department', help="Hospital department"),
                            'previous_antibiotics_exposure_cephalosporin': st.column_config.SelectboxColumn('Cephalosporin Exposure', help="Received cephalosporin?"),
                            'previous_antibiotics_exposure_carbapenem': st.column_config.SelectboxColumn('Carbapenem Exposure', help="Received carbapenem?"),
                            'previous_antibiotics_exposure_fluoroquinolone': st.column_config.SelectboxColumn('Fluoroquinolone Exposure', help="Received fluoroquinolone?"),
                            'previous_antibiotics_exposure_polymyxin': st.column_config.SelectboxColumn('Polymyxin Exposure', help="Received polymyxin?"),
                            'previous_antibiotics_exposure_aminoglycoside': st.column_config.SelectboxColumn('Aminoglycoside Exposure', help="Received aminoglycoside?"),
                            'previous_antibiotics_exposure_nitrofurantoin': st.column_config.SelectboxColumn('Nitrofurantoin Exposure', help="Received nitrofurantoin?"),
                            'previous_antibiotics_resistance_nitrofurantoin': st.column_config.SelectboxColumn('Nitrofurantoin Resistance', help="Past nitrofurantoin resistance?"),
                            'previous_antibiotics_resistance_sulfamethoxazole': st.column_config.SelectboxColumn('Sulfamethoxazole Resistance', help="Past sulfamethoxazole resistance?"),
                            'previous_antibiotics_resistance_ciprofloxacin': st.column_config.SelectboxColumn('Ciprofloxacin Resistance', help="Past ciprofloxacin resistance?"),
                            'previous_antibiotics_resistance_levofloxacin': st.column_config.SelectboxColumn('Levofloxacin Resistance', help="Past levofloxacin resistance?"),
                            'additional_note': st.column_config.TextColumn('Additional Note', help="Diagnosis or note text"),
                        }
                    )
                except Exception as e:
                    st.error(f"Could not load example file: {e}")

            with st.container():
                uploaded_file = st.file_uploader("📥 Upload your input CSV file", type=["csv"])
            if uploaded_file is not None:
                # Read uploaded file as DataFrame
                use_cols = ['age', 'race', 'veteran', 'gender', 'BMI',
                            'previous_antibiotics_exposure_cephalosporin',
                            'previous_antibiotics_exposure_carbapenem',
                            'previous_antibiotics_exposure_fluoroquinolone',
                            'previous_antibiotics_exposure_polymyxin',
                            'previous_antibiotics_exposure_aminoglycoside',
                            'previous_antibiotics_exposure_nitrofurantoin',
                            'previous_antibiotics_resistance_nitrofurantoin',
                            'previous_antibiotics_resistance_sulfamethoxazole',
                            'previous_antibiotics_resistance_ciprofloxacin',
                            'previous_antibiotics_resistance_levofloxacin', 'additional_note',
                            'organism_name', 'department_type', 'LOS',
                            'department_type', 'organism_name']
                final_df = pd.read_csv(uploaded_file, usecols=use_cols)
                final_df = final_df.rename(columns={'organism_name':'organism_name_before','department_type':'department_type_before'})

                st.write("🔍 Uploaded Data Preview:")
                st.dataframe(final_df.head())

                # Continue with prediction logic...
                    
        elif input_method == "Manual Input":
            # Create initial DataFrame for demographic information
            demo_data = {
                'age': [np.nan],
                'race': ['NA'],
                'veteran': ['NA'],
                'gender': ['NA'],
                'BMI': [np.nan],
            }

            hospitalization_data = {
                'department_type_before': ['NA'],
                'LOS': [np.nan],
            }

            # Create initial DataFrame for clinical information
            clinical_data = {
                'previous_antibiotics_exposure_cephalosporin': ['NA'],
                'previous_antibiotics_exposure_carbapenem': ['NA'],
                'previous_antibiotics_exposure_fluoroquinolone': ['NA'],
                'previous_antibiotics_exposure_polymyxin': ['NA'],
                'previous_antibiotics_exposure_aminoglycoside': ['NA'],
                'previous_antibiotics_exposure_nitrofurantoin': ['NA'],
                'previous_antibiotics_resistance_nitrofurantoin': ['NA'],
                'previous_antibiotics_resistance_sulfamethoxazole': ['NA'],
                'previous_antibiotics_resistance_ciprofloxacin': ['NA'],
                'previous_antibiotics_resistance_levofloxacin': ['NA'],
            }


            lab_data = {
                'organism_name_before': ['NA'],
            }


            # Create initial DataFrame for diagnosis notes
            diagnosis_data = {
                'additional_note': ['NA']
            }

            # Convert to DataFrames
            demo_df = pd.DataFrame(demo_data)
            clinical_df = pd.DataFrame(clinical_data)
            diagnosis_df = pd.DataFrame(diagnosis_data)
            lab_df = pd.DataFrame(lab_data)
            hospitalization_df = pd.DataFrame(hospitalization_data)

            # Set options for different columns
            options_bool = ['NA', True, False]
            options_gender = ['NA', 'Male', 'Female']
            options_dept = ['NA', 'ER', 'ICU', 'IP', 'OP']
            options_ogn = ['NA', 'Enterococcus faecium', 'Staphylococcus aureus', 
                        'Klebsiella pneumoniae', 'Acinetobacter baumannii',
                        'Pseudomonas aeruginosa', 'Enterobacter', 'Others']

            with st.container():
                # Input your demographic information
                st.markdown("<h6>🧑‍⚕️ Input your demographic information:</h6>", unsafe_allow_html=True)
                # Set index for demographic DataFrame
                edited_demo_df = st.data_editor(
                    demo_df, 
                    key='editor_demo',
                    num_rows="dynamic",
                    column_config={
                        'age': st.column_config.NumberColumn('Age', help="Enter the patient's age in years.", format="%d"),
                        'race': st.column_config.SelectboxColumn('Race (White/Non-White)', help="Select whether the patient is White or Non-White.", options=options_bool, default=None),
                        'veteran': st.column_config.SelectboxColumn('Veteran status', help="Indicate whether the patient has veteran status.", options=options_bool, default=None),
                        'gender': st.column_config.SelectboxColumn('Gender', help="Select the patient's gender.", options=options_gender, default=None),
                        'BMI': st.column_config.NumberColumn('BMI', help="Enter the patient's Body Mass Index (BMI), e.g., 23.45.", format="%.2f"),
                                        },
                    
                )

                # Input your clinical information
                st.markdown("<h6>🏥 Input your clinical hospitalization information:</h6>", unsafe_allow_html=True)
                # Set index for clinical DataFrame
                edited_hospitalization_df = st.data_editor(
                    hospitalization_df, 
                    key='editor_hospitalization',
                    num_rows="dynamic",
                    column_config={
                        'department_type_before': st.column_config.SelectboxColumn('Department', help="Select the hospital department the patient was admitted to.", options=options_dept, default=None),
                        'LOS': st.column_config.NumberColumn('LOS', help="Length of stay in the hospital (in days).", format="%d"),
                        # 'organism': st.column_config.SelectboxColumn('Organism', options=options_ogn, default=None),
                    }
                )
                # Input your clinical information
                st.markdown("<h6>💊 Input your antibiotics exposure and resistance history:</h6>", unsafe_allow_html=True)
                # Set index for clinical DataFrame
                edited_clinical_df = st.data_editor(
                    clinical_df, 
                    key='editor_clinical',
                    num_rows="dynamic",
                    column_config={
                        'previous_antibiotics_exposure_cephalosporin': st.column_config.SelectboxColumn('Previous Cephalosporin Exposure', help="Has the patient received cephalosporins in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_exposure_carbapenem': st.column_config.SelectboxColumn('Previous Carbapenem Exposure', help="Has the patient received carbapenems in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_exposure_fluoroquinolone': st.column_config.SelectboxColumn('Previous Fluoroquinolone Exposure', help="Has the patient received fluoroquinolones in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_exposure_polymyxin': st.column_config.SelectboxColumn('Previous Polymyxin Exposure', help="Has the patient received polymyxins in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_exposure_aminoglycoside': st.column_config.SelectboxColumn('Previous Aminoglycoside Exposure', help="Has the patient received aminoglycosides in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_exposure_nitrofurantoin': st.column_config.SelectboxColumn('Previous Nitrofurantoin Exposure', help="Has the patient received nitrofurantoin in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_resistance_nitrofurantoin': st.column_config.SelectboxColumn('Previous Nitrofurantoin Resistance', help="Was the patient infected with nitrofurantoin-resistant bacteria in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_resistance_sulfamethoxazole': st.column_config.SelectboxColumn('Previous Sulfamethoxazole Resistance', help="Was the patient infected with sulfamethoxazole-resistant bacteria in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_resistance_ciprofloxacin': st.column_config.SelectboxColumn('Previous Ciprofloxacin Resistance', help="Was the patient infected with ciprofloxacin-resistant bacteria in the past 90 days?", options=options_bool, default=None),
                        'previous_antibiotics_resistance_levofloxacin': st.column_config.SelectboxColumn('Previous Levofloxacin Resistance', help="Was the patient infected with levofloxacin-resistant bacteria in the past 90 days?", options=options_bool, default=None),
                    }
                )

                # Input your diagnosis notes
                st.markdown("<h6>📝 Input your diagnosis notes:</h6>", unsafe_allow_html=True)
                # Set index for diagnosis DataFrame
                edited_diagnosis_df = st.data_editor(
                    diagnosis_df, 
                    key='editor_diagnosis',
                    num_rows="dynamic",
                    column_config={
                        'additional_note': st.column_config.TextColumn('Additional Note', help="Enter any relevant clinical diagnosis or physician notes."),
                    }
                )

                st.markdown("<h6>🦠 Input your target organism(option):</h6>", unsafe_allow_html=True)
                edited_lab_df = st.data_editor(
                    lab_df, 
                    key='editor_lab',
                    num_rows="dynamic",
                    column_config={
                        'organism_name_before': st.column_config.SelectboxColumn('Target Organism', help="Select the organism for which resistance prediction is required (optional).", options=options_ogn, default=None),
                    }
                )

            # Combine edited DataFrames into one final DataFrame
            final_df = pd.concat([edited_demo_df, edited_clinical_df, edited_diagnosis_df,edited_lab_df, edited_hospitalization_df], axis=1)

            # Convert boolean columns
            to_correct_type = ['race', 'veteran', 'previous_antibiotics_exposure_cephalosporin','previous_antibiotics_exposure_carbapenem','previous_antibiotics_exposure_fluoroquinolone',
            'previous_antibiotics_exposure_polymyxin','previous_antibiotics_exposure_aminoglycoside','previous_antibiotics_exposure_nitrofurantoin','previous_antibiotics_resistance_ciprofloxacin',
            'previous_antibiotics_resistance_levofloxacin','previous_antibiotics_resistance_nitrofurantoin','previous_antibiotics_resistance_sulfamethoxazole' ]
            
            # todo gender
            for col in to_correct_type:
                print(final_df[col])
                final_df[col] = final_df[col].apply(convert_to_bool)  # 应用转换函数
                print(final_df[col])

            final_df['gender'] = final_df['gender'].apply(convert_gender_to_bool)  # 应用转换函数
            # Create boolean columns for dept and organism
            tempt_columns = {
                'department_type': final_df['department_type_before'].apply(convert_dept_to_num),
                'organism_name': final_df['organism_name_before'].apply(convert_organism_to_num),
            }

            

            # Convert to DataFrame and concatenate
            bool_df = pd.DataFrame(tempt_columns)

            # Final DataFrame for prediction
            final_df = pd.concat([final_df, bool_df], axis=1)
            # final_df = final_df.drop(columns=['dept', 'organism'])

        if 'selected_antibiotics' not in st.session_state:
            st.session_state.selected_antibiotics = []

        with st.container():
            selected_antibiotics = st.multiselect(
                "💊 Select antibiotics for resistance prediction:",
                list(antibiotics_options.keys())
        )
        
        st.session_state.selected_antibiotics = selected_antibiotics

        if st.button("Confirm and Predict"):
            st.session_state['predict_clicked'] = True

        # Add prediction button
        if st.session_state.get('predict_clicked', False):
            st.write(f"Start runing your task. Please Wait...")
            # final_df.to_csv('test11_before_preprocessing.csv',header=1, index=False)
            final_df = preprocess_data(final_df)
            final_df = final_df.fillna(-1)
            final_df = final_df.replace('NA', np.nan)
            X = final_df[feature_columns].fillna(-1).values

            final_df.to_csv('test11.csv',header=1, index=False)


            results_df = predict_with_ci(final_df, ['resistance_'+ v for v in selected_antibiotics],  )

            result_dir = '/ibex/user/xiex/ide/AMR_proj2/streamlit_running/tmp'
            results_csv_path = os.path.join(result_dir, 'combined_predictions_ver2.csv')
            results_df.to_csv(results_csv_path, index=False)
            st.session_state['results_df'] = results_df 

            if not results_df.empty:
                st.write(f"Prediction finished!")
                # Provide download option

                # Result Preview Section
                st.subheader("Result Preview:")
                with st.container():
                    st.dataframe(results_df.head(), use_container_width=True)  # Show the first few rows of results
                st.download_button(label="Download Predictions CSV", 
                        data=results_df.to_csv(index=False), 
                        file_name='combined_predictions.csv', 
                        mime='text/csv')
            st.subheader("Result Visualization:")
            plot_antibiotic_probabilities(results_csv_path, ['resistance_'+ v for v in selected_antibiotics])

    if not results_df.empty:
        with stylable_container(
            key="container_with_border_end",
            css_styles="""
                {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                width: 100%;  /* Ensure full width */
                max-width: 800px;  /* Set a max width to prevent overflow */
                margin: 0 auto;  /* Center the container */
                }
                """,
            ):
            st.markdown(
                """
                <h3>📈Analyze SHAP values for your uploaded sample:</h3>
                """,
                unsafe_allow_html=True
            )

            sample_indices = list(range(len(final_df)))  # Assuming final_df is your DataFrame with samples
            with st.container():
                selected_sample_index = st.selectbox("Select a sample index for SHAP analysis:", sample_indices)
                selected_antibiotic_sample = st.selectbox("Select antibiotic for sample SHAP analysis:", st.session_state.selected_antibiotics)


            if selected_sample_index is not None and selected_antibiotic_sample is not None:

                if st.button("Compute"):
                    if 'results_df' in st.session_state:
                        # Load the XGBoost model for the selected antibiotic
                        xgb_model = load_xgb_model(selected_antibiotic_sample)

                        # Prepare the feature matrix for the selected sample
                        X_sample = final_df.iloc[selected_sample_index][feature_columns].values.reshape(1, -1)

                        # Compute SHAP values for the specific sample
                        explainer = shap.Explainer(xgb_model)
                        shap_values = explainer(X_sample)
                        shap_values_rounded = np.round(shap_values.values, 3)

                        # Store results in session state
                        st.session_state.shap_values = shap_values_rounded
                        feature_names = feature_columns
                        
                        # Create the force plot
                        with st.container():
                            shap.force_plot(explainer.expected_value, shap_values_rounded, matplotlib=True, feature_names=feature_names)
                        plt.savefig('shap_force_plot.png')
                        
                        # Store image path in session state
                        st.session_state.shap_image_path = 'shap_force_plot.png'

                        # Create a DataFrame for feature scores
                        feature_scores_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'SHAP Score': shap_values_rounded.flatten()
                        })
                        
                        # Store DataFrame in session state
                        st.session_state.feature_scores_df = feature_scores_df

            # Check if results exist in session state and display them
            if 'shap_values' in st.session_state:
                st.markdown("<h4>SHAP Result:</h4>", unsafe_allow_html=True)
                st.markdown("<h6>SHAP Force Plot for Selected Sample:</h6>", unsafe_allow_html=True)
                with st.container():
                    st.image(st.session_state.shap_image_path)
                
                st.markdown("<h6>Feature Scores Table:</h6>", unsafe_allow_html=True)
                st.dataframe(st.session_state.feature_scores_df)

                csv = st.session_state.feature_scores_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Feature Scores",
                    data=csv,
                    file_name='feature_scores.csv',
                    mime='text/csv'
                )



# feature
with st.container():
    st.markdown(
        """
        <div class="block">
            <h3>⚠️ Disclaimer</h3>
            <p>
                This application is intended for research and educational purposes only. 
                It is not a substitute for professional medical advice, diagnosis, or treatment. 
                The predictions and results provided by this tool are generated based on machine learning models trained on historical electronic health record data and should not be solely relied upon to make clinical decisions.
            </p>
            <p>
                Always seek the advice of qualified healthcare professionals with any questions you may have regarding a medical condition or treatment decisions.
            </p>
            <p>
                By using this tool, you acknowledge that the developers are not responsible for any direct or indirect consequences arising from its use.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.button("Reset All"):
    st.session_state.clear()  # Clear all session state variables



