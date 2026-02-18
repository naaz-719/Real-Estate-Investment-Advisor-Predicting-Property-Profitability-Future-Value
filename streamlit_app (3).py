import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title='Real Estate Investment Advisor', layout='wide')

st.title('Real Estate Investment Advisor')
st.markdown('Interactive demo: filter properties, get investment classification, and estimate future price (5 years).')

# Load dataset if available
df = None
data_path = 'india_housing_prices.csv'
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.warning(f'Could not load dataset: {e}')

# Load models if they exist
clf = None
reg = None
if os.path.exists('investment_classifier.joblib'):
    try:
        clf = joblib.load('investment_classifier.joblib')
    except Exception as e:
        st.warning(f'Could not load classifier: {e}')
if os.path.exists('price_regressor.joblib'):
    try:
        reg = joblib.load('price_regressor.joblib')
    except Exception as e:
        st.warning(f'Could not load regressor: {e}')


#########################
# Sidebar filters & form
#########################
st.sidebar.header('Filter properties (dataset)')
if df is not None:
    min_size, max_size = int(df['Size_in_SqFt'].min()), int(df['Size_in_SqFt'].max())
    size_range = st.sidebar.slider('Size (SqFt)', min_value=min_size, max_value=max_size, value=(min_size, max_size))
    min_price, max_price = float(df['Price_in_Lakhs'].min()), float(df['Price_in_Lakhs'].max())
    price_range = st.sidebar.slider('Price (Lakhs)', min_value=min_price, max_value=max_price, value=(min_price, max_price))
    bhk_options = sorted(df['BHK'].dropna().unique().tolist())
    bhk_sel = st.sidebar.multiselect('BHK', options=bhk_options, default=bhk_options)
    state_options = sorted(df['State'].dropna().unique().tolist())
    state_sel = st.sidebar.multiselect('State (optional)', options=state_options, default=state_options)
else:
    size_range = (100, 5000)
    price_range = (0.0, 500.0)
    bhk_sel = []
    state_sel = []

st.sidebar.header('Manual input / Prediction')
with st.sidebar.form('single_input'):
    size = st.number_input('Size (SqFt)', min_value=1, value=1000)
    ppsf = st.number_input('Price per SqFt', min_value=0.0, value=0.05, format='%.3f')
    pta = st.slider('Public Transport Accessibility (1-10)', 1, 10, 5)
    schools = st.number_input('Nearby Schools', min_value=0, max_value=50, value=1)
    hospitals = st.number_input('Nearby Hospitals', min_value=0, max_value=50, value=1)
    amenities = st.number_input('Amenities count', min_value=0, max_value=20, value=2)
    ptype = st.selectbox('Property Type', ['Apartment','Independent House','Villa'])
    furn = st.selectbox('Furnished Status', ['Furnished','Semi-furnished','Unfurnished'])
    owner = st.selectbox('Owner Type', ['Owner','Builder','Broker'])
    facing = st.selectbox('Facing', ['North','South','East','West'])
    submitted = st.form_submit_button('Predict for these inputs')

#########################
# Main: show filtered table and visuals
#########################
st.header('Dataset view & filters')
if df is None:
    st.info('Dataset not available in workspace. Place `india_housing_prices.csv` in the project folder.')
else:
    filtered = df.copy()
    filtered = filtered[(filtered['Size_in_SqFt'] >= size_range[0]) & (filtered['Size_in_SqFt'] <= size_range[1])]
    filtered = filtered[(filtered['Price_in_Lakhs'] >= price_range[0]) & (filtered['Price_in_Lakhs'] <= price_range[1])]
    if len(bhk_sel) > 0:
        filtered = filtered[filtered['BHK'].isin(bhk_sel)]
    if len(state_sel) > 0:
        filtered = filtered[filtered['State'].isin(state_sel)]
    st.write(f'Showing {len(filtered)} properties matching filters')
    st.dataframe(filtered.head(200))

    # Visual 1: State vs average price_per_sqft bar
    st.subheader('Location insights')
    state_avg = filtered.groupby('State')['Price_per_SqFt'].mean().reset_index().sort_values('Price_per_SqFt', ascending=False)
    if not state_avg.empty:
        fig = px.bar(state_avg, x='State', y='Price_per_SqFt', title='Average Price per SqFt by State')
        st.plotly_chart(fig, use_container_width=True)

    # Visual 2: Heatmap of City vs Property_Type average price_per_sqft
    pivot = filtered.pivot_table(index='City', columns='Property_Type', values='Price_per_SqFt', aggfunc='mean').fillna(0)
    if not pivot.empty:
        fig2 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale='Viridis'))
        fig2.update_layout(title='Avg Price per SqFt: City vs Property Type', xaxis_title='Property Type', yaxis_title='City')
        st.plotly_chart(fig2, use_container_width=True)

    # Visual 3: Trend chart - top 5 localities by median price_per_sqft
    top_local = df.groupby('Locality')['Price_per_SqFt'].median().sort_values(ascending=False).head(5).index.tolist()
    if top_local:
        trend = df[df['Locality'].isin(top_local)].groupby(['Year_Built','Locality'])['Price_per_SqFt'].median().reset_index()
        fig3 = px.line(trend, x='Year_Built', y='Price_per_SqFt', color='Locality', markers=True, title='Price per SqFt over Years (Top 5 Localities)')
        st.plotly_chart(fig3, use_container_width=True)

#########################
# Predictions: single input
#########################
if submitted:
    X = pd.DataFrame([{ 
        'Size_in_SqFt': size,
        'Price_per_SqFt': ppsf,
        'Public_Transport_Accessibility': pta,
        'Nearby_Schools': schools,
        'Nearby_Hospitals': hospitals,
        'amenities_count': amenities,
        'Property_Type': ptype,
        'Furnished_Status': furn,
        'Owner_Type': owner,
        'Facing': facing
    }])

    st.subheader('Model outputs')
    # Classification: investment probability and decision
    if clf is not None:
        try:
            proba = clf.predict_proba(X)[0,1] if hasattr(clf, 'predict_proba') else float(clf.predict(X)[0])
            is_invest = 'Yes' if proba >= 0.5 else 'No'
            st.metric('Is this a good investment?', is_invest, delta=f'{proba:.2f} confidence')
            st.write(f'Investment probability: {proba:.3f}')

            # Feature importance (attempt)
            try:
                pre = clf.named_steps.get('preprocessor') if hasattr(clf, 'named_steps') else None
                model_step = clf.named_steps.get('classifier') if hasattr(clf, 'named_steps') else None
                if pre is not None and model_step is not None and hasattr(pre, 'get_feature_names_out'):
                    feat_names = pre.get_feature_names_out()
                    importances = model_step.feature_importances_
                    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(10)
                    st.subheader('Top feature importances (classifier)')
                    st.table(imp_df)
                else:
                    # Fallback: if model_step has feature_importances_ but no names
                    if model_step is not None and hasattr(model_step, 'feature_importances_'):
                        imps = model_step.feature_importances_
                        st.write('Feature importances available (array):')
                        st.write(imps)
            except Exception as e:
                st.info('Could not extract feature importances: ' + str(e))

        except Exception as e:
            st.error('Classifier prediction failed: ' + str(e))
    else:
        st.info('Classifier model not found. Run the notebook to train models and generate `investment_classifier.joblib`.')

    # Regression: predict current price and estimate after 5 years
    if reg is not None:
        try:
            price_now = reg.predict(X)[0]
            # compute annual growth rate from dataset if available
            if df is not None and 'Year_Built' in df.columns and 'Price_per_SqFt' in df.columns:
                try:
                    year_median = df.groupby('Year_Built')['Price_per_SqFt'].median().sort_index()
                    year_median = year_median[year_median.index.notna()]
                    # compute pct changes
                    pct = year_median.pct_change().dropna()
                    annual_growth = float(pct.mean()) if len(pct)>0 else 0.03
                except Exception:
                    annual_growth = 0.03
            else:
                annual_growth = 0.03
            price_5y = price_now * ((1 + annual_growth) ** 5)
            st.metric('Predicted current price (Lakhs)', f'{price_now:.2f}')
            st.metric('Estimated price after 5 years (Lakhs)', f'{price_5y:.2f}', delta=f'Assumed annual growth: {annual_growth:.3%}')
        except Exception as e:
            st.error('Regressor prediction failed: ' + str(e))
    else:
        st.info('Regressor model not found. Run the notebook to train models and generate `price_regressor.joblib`.')

#########################
# Batch upload & predictions
#########################
st.header('Batch predictions')
uploaded = st.file_uploader('Upload CSV for batch predictions (same columns as single input)', type=['csv'], key='batch')
if uploaded is not None:
    try:
        batch = pd.read_csv(uploaded)
        st.write('Preview:')
        st.dataframe(batch.head())
        required = ['Size_in_SqFt','Price_per_SqFt','Public_Transport_Accessibility','Nearby_Schools','Nearby_Hospitals','amenities_count','Property_Type','Furnished_Status','Owner_Type','Facing']
        missing = [c for c in required if c not in batch.columns]
        if missing:
            st.error('Missing columns: ' + ', '.join(missing))
        else:
            if clf is not None:
                try:
                    batch['investment_prob'] = clf.predict_proba(batch[required])[:,1] if hasattr(clf, 'predict_proba') else clf.predict(batch[required])
                except Exception as e:
                    st.error('Batch classifier failed: ' + str(e))
            if reg is not None:
                try:
                    batch['pred_price_lakhs'] = reg.predict(batch[required])
                    # estimate 5y
                    # reuse annual_growth computed above if possible
                    try:
                        year_median = df.groupby('Year_Built')['Price_per_SqFt'].median().sort_index()
                        pct = year_median.pct_change().dropna()
                        annual_growth = float(pct.mean()) if len(pct)>0 else 0.03
                    except Exception:
                        annual_growth = 0.03
                    batch['pred_price_5y'] = batch['pred_price_lakhs'] * ((1+annual_growth)**5)
                except Exception as e:
                    st.error('Batch regressor failed: ' + str(e))
            st.write('Results preview:')
            st.dataframe(batch.head())
            csv = batch.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions CSV', csv, file_name='predictions.csv', mime='text/csv')
    except Exception as e:
        st.error('Failed to read uploaded file: ' + str(e))

st.markdown('---')
st.write('Notes:')
st.write('- Train models by running `estate.ipynb` notebook. The Streamlit app expects `investment_classifier.joblib` and `price_regressor.joblib` in the same folder.')
st.write('- The 5-year estimate uses a simple dataset-derived average annual growth rate (fallback 3%). For production, use a proper time-series model or macroeconomic inputs.')

