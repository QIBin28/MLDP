import streamlit as st
import numpy as np
import pandas as pd
import joblib


model = joblib.load('model')



towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
         'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
         'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
         'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
         'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
         'TOA PAYOH', 'WOODLANDS', 'YISHUN']

flat_types = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM',
              'MULTI-GENERATION']

storey_ranges = ['10 TO 12', '01 TO 03', '04 TO 06', '07 TO 09', '13 TO 15',
                 '19 TO 21', '22 TO 24', '16 TO 18', '34 TO 36', '28 TO 30',
                 '37 TO 39', '49 TO 51', '25 TO 27', '40 TO 42', '31 TO 33',
                 '46 TO 48', '43 TO 45']

st.title("HDB Resale Price Prediction")

town_selected = st.selectbox("select Town",towns)
flat_types_selected = st.selectbox("select Flat Type",flat_types)
storney_range_selected = st.selectbox("select Storey Range",storey_ranges)
floor_area_selected = st.slider("select Floor Area(spm)", min_value=30, max_value=200, )

if st.button("Predict Price"):
    input_data = {
        "town": [town_selected],
        "flat_type": [flat_types_selected],
        "storey_range": [storney_range_selected],
        "floor_area_sqm": [floor_area_selected]
    }

    df_input = pd.DataFrame({'town': [selected_town],
                            'flat_type': [selected_flat_type],
                            'storey_range': [selected_storey_range],
                            'floor_area_sqm': [selected_floor_area_sqm]})
    input_df = pd.get_dummies(input_df, columns=['town', 'flat_type', 'storey_range'])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")


st.markdown(
    f"""
    <style>
    .stApp((
    background:url('https://sg.images.search.yahoo.com/images/view;_ylt=Awr1RddO5W1obXMPIQMl4gt.;_ylu=c2VjA3NyBHNsawNpbWcEb2lkA2M0NzYwYWU5ZWI3MGRkMjE2MDA4NGZjNjVkZjY5M2RhBGdwb3MDMjMEaXQDYmluZw--?back=https%3A%2F%2Fsg.images.search.yahoo.com%2Fsearch%2Fimages%3Fp%3Dcute%2Bimage%26type%3DE210SG885G0%26fr%3Dmcafee%26fr2%3Dpiv-web%26tab%3Dorganic%26ri%3D23&w=3000&h=3000&imgurl=i.etsystatic.com%2F43739964%2Fr%2Fil%2Fb6f0d0%2F5081398183%2Fil_fullxfull.5081398183_7p82.jpg&rurl=https%3A%2F%2Fwww.etsy.com%2Fca%2Flisting%2F1507792039%2Fanime-girl-cute-girl-happy-kawaii&size=1098KB&p=cute+image&oid=c4760ae9eb70dd2160084fc65df693da&fr2=piv-web&fr=mcafee&tt=Anime+Girl%2C+Cute+Girl%2C+Happy%2C+Kawaii%2C+Digital+Art%2C+Anime%2C+Otaku%2C+Anime+...&b=0&ni=21&no=23&ts=&tab=organic&sigr=FCGdOQfZNk7c&sigb=rzS95zBM4_6C&sigi=AiYj0kjCbR7M&sigt=FpvouHkK_YzX&.crumb=mStxwO8ehLJ&fr=mcafee&fr2=piv-web&type=E210SG885G0');
    background-size: cover;
    ))
    </style>
    """,
    unsafe_allow_html=True
)