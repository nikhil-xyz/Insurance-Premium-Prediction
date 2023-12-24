import pickle
import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.title('Insurance Premium Prediction')

def callback_search():
    #   Destroying old sessions
    st.session_state['search_btn'] = False

# Extracting features from the webpage
# capturing the selected age
option_age = st.selectbox(
        'select the age',
        [i for i in range(15, 65)],
        on_change=callback_search
    )
# print(option_age)

# capturing the selected gender
option_gender = st.selectbox(
        'select the gender',
        ['female', 'male'],
        on_change=callback_search
    )
# print(option_gender)

# capturing the bmi
option_bmi = st.text_input(
        'enter the bmi (values should be in range 15-60)',
        on_change=callback_search
    )
# print(option_bmi)

# capturing the number of children
option_children = st.selectbox(
        'select number of children',
        [i for i in range(6)],
        on_change=callback_search
    )
# print(option_children)

# capturing the smoker
option_smoker = st.selectbox(
        'is the candidate smoker',
        ['yes', 'no'],
        on_change=callback_search
    )
# print(option_smoker)

# capturing the region
option_region = st.selectbox(
        'select the region',
        ['southwest', 'southeast', 'northwest', 'northeast'],
        on_change=callback_search
    )
# print(option_region)

data = CustomData(option_age, option_gender, option_bmi, option_children, option_smoker, option_region)
pred_df = data.get_data_as_dataframe()
predict_pipeline = PredictPipeline()

prediction = st.button('Predict')
# adding session state fot 'prediction' button
if st.session_state.get('search_btn') != True:
    st.session_state['search_btn'] = prediction

# prediction and printing the output
if st.session_state['search_btn']:
    st.write('The Predicted Expense is: ')
    # print(predict_pipeline.predict(pred_df))
    st.write(predict_pipeline.predict(pred_df))

