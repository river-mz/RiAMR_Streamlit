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
from utils import preprocess_data
import shap 
import matplotlib.pyplot as plt

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
    "nitrofurantoin": {
        "lr_model": "lr_model_resistance_nitrofurantoin.pth",
        "xgb_model": "xgb_model_resistance_nitrofurantoin.pkl"
    },
    "sulfamethoxazole": {
        "lr_model": "lr_model_resistance_sulfamethoxazole.pth",
        "xgb_model": "xgb_model_resistance_sulfamethoxazole.pkl"
    },
    "ciprofloxacin": {
        "lr_model": "lr_model_resistance_ciprofloxacin.pth",
        "xgb_model": "xgb_model_resistance_ciprofloxacin.pkl"
    },
    "levofloxacin": {
        "lr_model": "lr_model_resistance_levofloxacin.pth",
        "xgb_model": "xgb_model_resistance_levofloxacin.pkl"
    }
}


# Define model directory
model_dir = '/home/linp0a/AMR_prediction_pipeline/model_prediction/model_Ours_Oct_5'


# Load the feature columns from the JSON file
with open('feature_columns.json', 'r') as file:
    feature_columns = json.load(file)

results_df = pd.DataFrame()



def convert_to_bool(value):
    if isinstance(value, str):  # 检查是否为字符串
        if value.lower() == "true":  # 将 "true" 转换为 True
            return True
        elif value.lower() == "false":  # 将 "false" 转换为 False
            return False
    return np.nan  # 对于其他情况，返回布尔值


@st.cache_resource  # Cache the model loading function
def load_xgb_model(antibiotic):
    model_path = os.path.join(model_dir, antibiotics_options[antibiotic]["xgb_model"])
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Function to display demo DataFrame
def show_demo_data():
    demo_df = pd.read_csv('/home/linp0a/AMR_prediction_pipeline/test_demo_input3.csv')
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
    st.title("RiAMR -- Robust and Interpretable Antimicrobial Resistance Prediction from Electronic Health Records")
    st.markdown(
        """
        <h4>Welcome to the AMR Prediction Application!</h4>
        <p>This application is designed to assist healthcare professionals in predicting antimicrobial resistance (AMR) based on patient data. 
        By leveraging advanced machine learning models, it provides insights into the likelihood of resistance to various antibiotics, enabling better-informed treatment decisions.
        </p>
        """,
        unsafe_allow_html=True
    )

    # feature
    with st.container():
        st.markdown(
            """
            <div class="block">
            <h3>Key Features:</h3>
            <ul class="features">
                <li><strong>User-Friendly Interface:</strong> Easily upload your patient data in CSV format or input it manually.</li>
                <li><strong>Model Selection:</strong> Choose from multiple antibiotics for resistance prediction, including nitrofurantoin, sulfamethoxazole, ciprofloxacin, and levofloxacin.</li>
                <li><strong>SHAP Visualization:</strong> Understand the model's predictions through SHAP (SHapley Additive exPlanations) values, which highlight the contribution of each feature to the prediction.</li>
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
            <h3>How the Model Makes Predictions</h3>
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
            global_shap_image_path = f'/home/linp0a/AMR_prediction_pipeline/model_prediction/shap_Ours_Oct_19/shap_xgb_resistance_{selected_antibiotic_global}.png'
            st.image(global_shap_image_path, caption=f'Global SHAP value in XGBoost for {selected_antibiotic_global}', width=620)

elif selected == "Predict":
    st.title("RiAMR -- Robust and Interpretable Antimicrobial Resistance Prediction from Electronic Health Records")


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
            # Upload file section
            with st.container():
                uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
            if uploaded_file is not None:
                # Read uploaded file as DataFrame
                use_cols = ['age', 'race', 'veteran', 'gender', 'BMI', 
                            'previous_antibiotic_exposure_cephalosporin',
                            'previous_antibiotic_exposure_carbapenem',
                            'previous_antibiotic_exposure_fluoroquinolone',
                            'previous_antibiotic_exposure_polymyxin',
                            'previous_antibiotic_exposure_aminoglycoside',
                            'previous_antibiotic_exposure_nitrofurantoin',
                            'previous_antibiotic_resistance_ciprofloxacin',
                            'previous_antibiotic_resistance_levofloxacin',
                            'previous_antibiotic_resistance_nitrofurantoin',
                            'previous_antibiotic_resistance_sulfamethoxazole',
                            'dept_ER', 'dept_ICU', 'dept_IP',
                            'dept_OP', 'dept_nan', 'Enterococcus_faecium',
                            'Staphylococcus_aureus', 'Klebsiella_pneumoniae',
                            'Acinetobacter_baumannii', 'Pseudomonas_aeruginosa',
                            'Enterobacter', 'organism_other', 'organism_NA', 'additional_note'] 
                
                final_df = pd.read_csv(uploaded_file, usecols=use_cols)
                
                st.write("Uploaded Data Preview:")
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

            # Create initial DataFrame for clinical information
            clinical_data = {
                'previous_antibiotic_exposure_cephalosporin': ['NA'],
                'previous_antibiotic_exposure_carbapenem': ['NA'],
                'previous_antibiotic_exposure_fluoroquinolone': ['NA'],
                'previous_antibiotic_exposure_polymyxin': ['NA'],
                'previous_antibiotic_exposure_aminoglycoside': ['NA'],
                'previous_antibiotic_exposure_nitrofurantoin': ['NA'],
                'previous_antibiotic_resistance_nitrofurantoin': ['NA'],
                'previous_antibiotic_resistance_sulfamethoxazole': ['NA'],
                'previous_antibiotic_resistance_ciprofloxacin': ['NA'],
                'previous_antibiotic_resistance_levofloxacin': ['NA'],
                'dept': ['NA'],
                'organism': ['NA'],
            }

            # Create initial DataFrame for diagnosis notes
            diagnosis_data = {
                'additional_note': ['NA']
            }

            # Convert to DataFrames
            demo_df = pd.DataFrame(demo_data)
            clinical_df = pd.DataFrame(clinical_data)
            diagnosis_df = pd.DataFrame(diagnosis_data)

            # Set options for different columns
            options_bool = ['NA', True, False]
            options_gender = ['NA', 'Male', 'Female']
            options_dept = ['NA', 'ER', 'ICU', 'IP', 'OP']
            options_ogn = ['NA', 'Enterococcus faecium', 'Staphylococcus aureus', 
                        'Klebsiella pneumoniae', 'Acinetobacter baumannii',
                        'Pseudomonas aeruginosa', 'Enterobacter', 'Others']

            with st.container():
                # Input your demographic information
                st.markdown("<h6>Input your demographic information:</h6>", unsafe_allow_html=True)
                # Set index for demographic DataFrame
                edited_demo_df = st.data_editor(
                    demo_df, 
                    key='editor_demo',
                    num_rows="dynamic",
                    column_config={
                        'age': st.column_config.NumberColumn('Age', help="Input the patient's age", format="%d"),
                        'race': st.column_config.SelectboxColumn('Race (White/Non-White)', options=options_bool, default=None),
                        'veteran': st.column_config.SelectboxColumn('Veteran status', options=options_bool, default=None),
                        'gender': st.column_config.SelectboxColumn('Gender', options=options_gender, default=None),
                        'BMI': st.column_config.NumberColumn('BMI', help="Input the patient's BMI", format="%.2f"),
                    },
                    
                )


                # Input your clinical information
                st.markdown("<h6>Input your clinical information:</h6>", unsafe_allow_html=True)
                # Set index for clinical DataFrame
                edited_clinical_df = st.data_editor(
                    clinical_df, 
                    key='editor_clinical',
                    num_rows="dynamic",
                    column_config={
                        'previous_antibiotic_exposure_cephalosporin': st.column_config.SelectboxColumn('Previous Cephalosporin Exposure', options=options_bool, default=None),
                        'previous_antibiotic_exposure_carbapenem': st.column_config.SelectboxColumn('Previous Carbapenem Exposure', options=options_bool, default=None),
                        'previous_antibiotic_exposure_fluoroquinolone': st.column_config.SelectboxColumn('Previous Fluoroquinolone Exposure', options=options_bool, default=None),
                        'previous_antibiotic_exposure_polymyxin': st.column_config.SelectboxColumn('Previous Polymyxin Exposure', options=options_bool, default=None),
                        'previous_antibiotic_exposure_aminoglycoside': st.column_config.SelectboxColumn('Previous Aminoglycoside Exposure', options=options_bool, default=None),
                        'previous_antibiotic_exposure_nitrofurantoin': st.column_config.SelectboxColumn('Previous Nitrofurantoin Exposure', options=options_bool, default=None),
                        'previous_antibiotic_resistance_nitrofurantoin': st.column_config.SelectboxColumn('Previous Nitrofurantoin Resistance', options=options_bool, default=None),
                        'previous_antibiotic_resistance_sulfamethoxazole': st.column_config.SelectboxColumn('Previous Sulfamethoxazole Resistance', options=options_bool, default=None),
                        'previous_antibiotic_resistance_ciprofloxacin': st.column_config.SelectboxColumn('Previous Ciprofloxacin Resistance', options=options_bool, default=None),
                        'previous_antibiotic_resistance_levofloxacin': st.column_config.SelectboxColumn('Previous Levofloxacin Resistance', options=options_bool, default=None),
                        'dept': st.column_config.SelectboxColumn('Department', options=options_dept, default=None),
                        'organism': st.column_config.SelectboxColumn('Organism', options=options_ogn, default=None),
                    }
                )

                # Input your diagnosis notes
                st.markdown("<h6>Input your diagnosis notes:</h6>", unsafe_allow_html=True)
                # Set index for diagnosis DataFrame
                edited_diagnosis_df = st.data_editor(
                    diagnosis_df, 
                    key='editor_diagnosis',
                    num_rows="dynamic",
                    column_config={
                        'additional_note': st.column_config.TextColumn('Additional Note', default=None)
                    }
                )

            # Combine edited DataFrames into one final DataFrame
            final_df = pd.concat([edited_demo_df, edited_clinical_df, edited_diagnosis_df], axis=1)

            # Convert boolean columns
            to_correct_type = ['race', 'veteran', 'gender', 'previous_antibiotic_exposure_cephalosporin','previous_antibiotic_exposure_carbapenem','previous_antibiotic_exposure_fluoroquinolone',
            'previous_antibiotic_exposure_polymyxin','previous_antibiotic_exposure_aminoglycoside','previous_antibiotic_exposure_nitrofurantoin','previous_antibiotic_resistance_ciprofloxacin',
            'previous_antibiotic_resistance_levofloxacin','previous_antibiotic_resistance_nitrofurantoin','previous_antibiotic_resistance_sulfamethoxazole' ]
            
            for col in to_correct_type:
                final_df[col] = final_df[col].apply(convert_to_bool)  # 应用转换函数

            # Create boolean columns for dept and organism
            bool_columns = {
                'dept_ER': final_df['dept'].apply(lambda x: x == 'ER'),
                'dept_ICU': final_df['dept'].apply(lambda x: x == 'ICU'),
                'dept_IP': final_df['dept'].apply(lambda x: x == 'IP'),
                'dept_OP': final_df['dept'].apply(lambda x: x == 'OP'),
                'dept_nan': final_df['dept'].apply(lambda x: x == 'NA'),
                
                'Enterococcus_faecium': final_df['organism'].apply(lambda x: x == 'Enterococcus faecium'),
                'Staphylococcus_aureus': final_df['organism'].apply(lambda x: x == 'Staphylococcus aureus'),
                'Klebsiella_pneumoniae': final_df['organism'].apply(lambda x: x == 'Klebsiella pneumoniae'),
                'Acinetobacter_baumannii': final_df['organism'].apply(lambda x: x == 'Acinetobacter baumannii'),
                'Pseudomonas_aeruginosa': final_df['organism'].apply(lambda x: x == 'Pseudomonas aeruginosa'),
                'Enterobacter': final_df['organism'].apply(lambda x: x == 'Enterobacter'),
                'organism_other': final_df['organism'].apply(lambda x: x == 'Others'),
                'organism_NA': final_df['organism'].apply(lambda x: x == 'NA'),
            }

            # Convert to DataFrame and concatenate
            bool_df = pd.DataFrame(bool_columns)

            # Final DataFrame for prediction
            final_df = pd.concat([final_df, bool_df], axis=1)
            final_df = final_df.drop(columns=['dept', 'organism'])

        if 'selected_antibiotics' not in st.session_state:
            st.session_state.selected_antibiotics = []

        with st.container():
            selected_antibiotics = st.multiselect(
                "Select antibiotics for resistance prediction:",
                list(antibiotics_options.keys())
        )
        
        st.session_state.selected_antibiotics = selected_antibiotics

        if st.button("Confirm and Predict"):
            st.session_state['predict_clicked'] = True

        # Add prediction button
        if st.session_state.get('predict_clicked', False):
            st.write(f"Start runing your task. Please Wait...")
            final_df = preprocess_data(final_df)
            final_df = final_df.fillna(-1)
            final_df = final_df.replace('NA', np.nan)
            X = final_df[feature_columns].fillna(-1).values

            final_df.to_csv('test11.csv',header=1)

            for antibiotic in selected_antibiotics:
                st.write(f"Predicting resistance for {antibiotic}...")


                xgb_model = load_xgb_model(antibiotic)

                xgb_proba = xgb_model.predict_proba(X)[:, 1]
                # print('run xgb finished')

                results_df[f'{antibiotic}_resistance_xgboost_proba'] = np.round(xgb_proba.flatten(), 3)
            
            results_csv_path = os.path.join(model_dir, 'combined_predictions.csv')
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
                


    if not results_df.empty:
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
                <h3>Analyze SHAP values for your uploaded sample:</h3>
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


if st.button("Reset All"):
    st.session_state.clear()  # Clear all session state variables



