import folium
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import calendar
import json
import geopandas as gpd
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error,
    median_absolute_error, accuracy_score, confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

class Config:
    PAGE_TITLE = "Acute Malnutrition Early Warning System"
    PAGE_LAYOUT = "wide"
    RISK_COLORS = {
        "Normal": "#2ecc71",
        "Watch": "#f1c40f",
        "Alert": "#e67e22",
        "Emergency": "#e74c3c",
        "No Data": "#95a5a6"
    }
    CURATED_FEATURES = {
        'target_lag_1': 'Immediate',
        'target_ma3': '3-month trend',
        'target_lag_3': 'Seasonal cycles',
        'target_lag_6': 'Long-term patterns',
        'col_108_CD01a_Diarrhea_Acute_Cases_ma3': 'Diarrhea (3-month avg)',
        'col_105_EP01c_Malaria_confirmed_Blood_smear_and_RDT': 'Malaria cases',
        'col_107a_PP01_Projected_Population': 'Population',
        'Average_of_mean_gpp': 'Vegetation/GPP',
        'CCH_Precipitation_CHIRPS_lag_3': 'Rainfall (3-month lag)',
        'CCH_Air_temperature_ERA5_Land_ma3': 'Temperature (3-month avg)',
        'month_sin': 'Seasonal pattern',
        'month_cos': 'Seasonal pattern',
        'quarter': 'Quarter',
        'month': 'Month'
    }
    MODEL_CONFIG = {
        'model': RandomForestRegressor,
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'max_features': 'sqrt'
        }
    }
    TRAIN_TEST_SPLIT = 0.75
    CONFIDENCE_INTERVAL = 1.96


def classify_risk_seasonal(admissions, month_mean, month_std):
    if pd.isna(admissions) or pd.isna(month_mean) or pd.isna(month_std):
        return "No Data"

    if month_std <= 0:
        if admissions <= month_mean:
            return "Normal"
        elif admissions <= month_mean * 1.3:
            return "Watch"
        elif admissions <= month_mean * 1.6:
            return "Alert"
        else:
            return "Emergency"

    sd_above_mean = (admissions - month_mean) / month_std

    if sd_above_mean > 2.0:
        return "Emergency"
    elif sd_above_mean > 1.0:
        return "Alert"
    elif sd_above_mean > 0.5:
        return "Watch"
    else:
        return "Normal"


def clean_column_names(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.replace(r'[\s\.\/\-]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
        .str.replace(r'^(\d+)', r'col_\1', regex=True)
        .str.replace(r'[\(\)]', '', regex=True)
        .str.replace(r'\&', 'and', regex=True)
        .str.replace(r'B_s_', 'Blood_smear_', regex=True)
    )
    return df


def parse_date(date_str):
    try:
        parts = str(date_str).split()
        month_name = parts[0].capitalize()
        year = int(parts[1])
        month_num = list(calendar.month_name).index(month_name)
        return pd.Timestamp(year=year, month=month_num, day=1)
    except:
        return pd.to_datetime(date_str, errors='coerce')


def read_csv_with_retry(file):
    try:
        content = file.read()
        if hasattr(file, 'seek'):
            file.seek(0)
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        try:
            df = pd.read_csv(StringIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(content), encoding='latin-1')
        return df
    except:
        if hasattr(file, 'seek'):
            file.seek(0)
        df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
        return df


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    @st.cache_data
    def load_and_prepare_data(_self, file):  # <- leading underscore
        df = read_csv_with_retry(file)
        if df.empty:
            raise ValueError("Uploaded file is empty.")

        df = clean_column_names(df)

        required_columns = ['periodname', 'Total_Admissions', 'organisationunitname']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df['Date'] = df['periodname'].apply(parse_date)
        df = df.dropna(subset=['Date', 'Total_Admissions']).sort_values(['organisationunitname', 'Date'])
        df['target'] = np.log1p(df['Total_Admissions'])

        # Add seasonal statistics
        df = _self._calculate_seasonal_statistics(df)
        df = _self._add_seasonal_features(df)
        features_dict = _self._get_curated_features(df)

        return df.dropna(subset=['target']), features_dict, list(features_dict.keys())

    def _calculate_seasonal_statistics(self, df):
        df['month_mean'] = np.nan
        df['month_std'] = np.nan
        df['actual_risk'] = "No Data"

        for district in df['organisationunitname'].unique():
            district_df = df[df['organisationunitname'] == district].sort_values('Date')
            for month in range(1, 13):
                month_mask = district_df['Date'].dt.month == month
                if not month_mask.any():
                    continue
                admissions = district_df.loc[month_mask, 'Total_Admissions']
                indices = district_df[month_mask].index
                for i, idx in enumerate(indices):
                    if i < 2:
                        mean_val = np.nan
                        std_val = np.nan
                    else:
                        mean_val = admissions.iloc[:i].mean()
                        std_val = admissions.iloc[:i].std()
                    df.loc[idx, 'month_mean'] = mean_val
                    df.loc[idx, 'month_std'] = std_val
                    if not pd.isna(mean_val) and not pd.isna(std_val):
                        df.loc[idx, 'actual_risk'] = classify_risk_seasonal(admissions.iloc[i], mean_val, std_val)

        df['month_mean'].fillna(df['Total_Admissions'].mean(), inplace=True)
        df['month_std'].fillna(df['Total_Admissions'].std(), inplace=True)
        mask = df['actual_risk'] == "No Data"
        df.loc[mask, 'actual_risk'] = df[mask].apply(
            lambda r: classify_risk_seasonal(r['Total_Admissions'], r['month_mean'], r['month_std']), axis=1
        )
        return df

    @staticmethod
    def _add_seasonal_features(df):
        df['month'] = df['Date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        return df

    def _get_curated_features(self, df):
        features = {}
        for lag in [1, 3, 6]:
            name = f'target_lag_{lag}'
            df[name] = df.groupby('organisationunitname')['target'].shift(lag)
            features[name] = Config.CURATED_FEATURES.get(name, name)

        df['target_ma3'] = df.groupby('organisationunitname')['target'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=2).mean()
        )
        features['target_ma3'] = Config.CURATED_FEATURES.get('target_ma3', '3-month target average')

        for col, desc in [
            ('col_108_CD01a_Diarrhea_Acute_Cases', 'Diarrhea (3-month avg)'),
            ('col_105_EP01c_Malaria_confirmed_Blood_smear_and_RDT', 'Malaria cases'),
            ('col_107a_PP01_Projected_Population', 'Population'),
            ('Average_of_mean_gpp', 'Vegetation/GPP'),
            ('CCH_Precipitation_CHIRPS', 'Rainfall (3-month lag)'),
            ('CCH_Air_temperature_ERA5_Land', 'Temperature (3-month avg)')
        ]:
            if col in df.columns:
                df[f'{col}_ma3'] = df.groupby('organisationunitname')[col].transform(
                    lambda x: x.shift(1).rolling(3, min_periods=2).mean()
                )
                features[f'{col}_ma3'] = desc

        for f in ['month_sin', 'month_cos', 'quarter', 'month']:
            features[f] = Config.CURATED_FEATURES.get(f, f)
            df[f] = df[f].fillna(0)

        return features


class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train_and_forecast(self, df, features_dict, horizon=3):
        features = list(features_dict.keys())
        df = df.dropna(subset=['target'])
        df[features] = df[features].fillna(0)

        split = int(len(df) * self.config.TRAIN_TEST_SPLIT)
        train = df.iloc[:split]
        test = df.iloc[split:]

        model_class = self.config.MODEL_CONFIG['model']
        model = model_class(**self.config.MODEL_CONFIG['params'])
        scaler = RobustScaler()

        X_train = scaler.fit_transform(train[features])
        y_train = train['target']
        X_test = scaler.transform(test[features])
        y_test = test['target']

        model.fit(X_train, y_train)
        test_pred_log = model.predict(X_test)
        test_pred = np.expm1(test_pred_log)
        actual = np.expm1(y_test)

        metrics = {
            'r2': r2_score(actual, test_pred),
            'mae': mean_absolute_error(actual, test_pred),
            'rmse': np.sqrt(mean_squared_error(actual, test_pred)),
            'median_ae': median_absolute_error(actual, test_pred),
            'pred_std': np.std(actual - test_pred)
        }

        # Risk evaluation
        pred_risks = [
            classify_risk_seasonal(p, m, s)
            for p, m, s in zip(test_pred, test['month_mean'], test['month_std'])
        ]
        actual_risks = test['actual_risk'].values
        labels = ['Normal', 'Watch', 'Alert', 'Emergency']
        cm = confusion_matrix(actual_risks, pred_risks, labels=labels)
        accuracy = accuracy_score(actual_risks, pred_risks)

        result = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'risk_labels': labels,
            'test_pred': test_pred,
            'actual': actual
        }

        # Forecast
        forecast_df = self._generate_forecasts(df, result, features, horizon)
        return result, forecast_df

    def _generate_forecasts(self, df, result, features, horizon):
        forecasts = []
        last_date = df['Date'].max()
        forecast_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon+1)]
        model = result['model']
        scaler = result['scaler']
        pred_std = result['metrics']['pred_std']

        for district in df['organisationunitname'].unique():
            district_df = df[df['organisationunitname'] == district].sort_values('Date')
            working_df = district_df.copy()

            for forecast_date in forecast_dates:
                new_row = working_df.iloc[-1:].copy()
                new_row['Date'] = forecast_date
                month = forecast_date.month
                new_row['month'] = month
                new_row['month_sin'] = np.sin(2*np.pi*month/12)
                new_row['month_cos'] = np.cos(2*np.pi*month/12)
                new_row['quarter'] = (month-1)//3 +1
                new_row['year'] = forecast_date.year
                for lag in [1,3,6]:
                    col = f'target_lag_{lag}'
                    new_row[col] = working_df['target'].iloc[-lag] if len(working_df)>=lag else 0
                new_row['target_ma3'] = working_df['target'].iloc[-3:].mean() if len(working_df)>=3 else 0

                working_df = pd.concat([working_df, new_row], ignore_index=True)
                X = scaler.transform(new_row[features].fillna(0).values.reshape(1,-1))
                pred_log = model.predict(X)[0]
                prediction = max(0, np.expm1(pred_log))

                hist = df[(df['organisationunitname']==district) & (df['Date'].dt.month==month)]
                month_mean = hist['Total_Admissions'].mean() if not hist.empty else df['Total_Admissions'].mean()
                month_std = hist['Total_Admissions'].std() if not hist.empty else df['Total_Admissions'].std()
                risk = classify_risk_seasonal(prediction, month_mean, month_std)

                forecasts.append({
                    'District': district,
                    'Date': forecast_date,
                    'Month_Year': forecast_date.strftime('%B %Y'),
                    'Predicted': round(prediction),
                    'Lower_95CI': round(max(0, prediction - Config.CONFIDENCE_INTERVAL*pred_std)),
                    'Upper_95CI': round(prediction + Config.CONFIDENCE_INTERVAL*pred_std),
                    'Risk': risk,
                    'Month_Mean': round(month_mean,1),
                    'Month_STD': round(month_std,1),
                    'SD_Above_Mean': round((prediction - month_mean)/month_std,2) if month_std>0 else 0
                })

        return pd.DataFrame(forecasts)


def main():
    st.set_page_config(page_title=Config.PAGE_TITLE, layout=Config.PAGE_LAYOUT)
    st.title("Acute Malnutrition Early Warning System")
    st.caption("Seasonal Risk Classification – Curated Feature Set")

    data_file = st.sidebar.file_uploader("Upload CSV data", type="csv")
    geo_file = st.sidebar.file_uploader("Upload GeoJSON for risk map", type=["json","geojson"])
    horizon = 3

    st.sidebar.info(
        f"**Risk Classification:**\n\n"
        f"- **Normal**: ≤ monthly mean\n"
        f"- **Watch**: > mean + 0.5 σ\n"
        f"- **Alert**: > mean + 1.0 σ\n"
        f"- **Emergency**: > mean + 2.0 σ"
    )

    if data_file:
        try:
            processor = DataProcessor(Config())
            df, features_dict, features = processor.load_and_prepare_data(data_file)

            # Display raw data safely
            df_display = df.copy()
            for col in df_display.columns:
                if pd.api.types.is_datetime64_any_dtype(df_display[col]):
                    df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
            st.dataframe(df_display, use_container_width=True)

            # Train model
            with st.spinner("Training model..."):
                trainer = ModelTrainer(Config())
                result, forecast_df = trainer.train_and_forecast(df, features_dict, horizon)

            if forecast_df is not None and not forecast_df.empty:
                forecast_df['Month_Year'] = forecast_df['Date'].dt.strftime('%B %Y')
                forecast_df['Month_Num'] = forecast_df['Date'].dt.month
                forecast_df['Year'] = forecast_df['Date'].dt.year
                forecast_df = forecast_df.sort_values(['Year','Month_Num'])
                forecast_months = forecast_df['Month_Year'].unique()
                st.success(f"Forecast generated for: {', '.join(forecast_months)}")

                # Tabs
                tab0, tab1, tab2, tab3, tab4 = st.tabs([
                    "Descriptive Analysis", "Performance", "Forecasts", "Risk Maps", "Feature Importance"
                ])

                # ═════════ Tab 0 ═════════
                with tab0:
                    st.subheader("Descriptive Analysis")
                    st.markdown("### Data Summary")
                    st.dataframe(df.describe(), use_container_width=True)

                    st.markdown("### Correlations")
                    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    corr_pearson = df[numerical_cols].corr(method='pearson')
                    corr_spearman = df[numerical_cols].corr(method='spearman')

                    fig_pearson = px.imshow(corr_pearson, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
                    fig_pearson.update_layout(title="Pearson Correlation Heatmap", height=800)
                    st.plotly_chart(fig_pearson, use_container_width=True)

                    fig_spearman = px.imshow(corr_spearman, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r')
                    fig_spearman.update_layout(title="Spearman Correlation Heatmap", height=800)
                    st.plotly_chart(fig_spearman, use_container_width=True)

                    st.markdown("### Visual Relationships")
                    districts = sorted(df['organisationunitname'].unique())
                    selected_district = st.selectbox("Select District for Plots:", districts, key="desc_district")
                    district_df = df[df['organisationunitname']==selected_district].sort_values('Date')

                    def normalize(series):
                        if series.max() - series.min() == 0:
                            return series
                        return (series - series.min()) / (series.max() - series.min())

                    fig_rel = go.Figure()
                    fig_rel.add_trace(go.Scatter(
                        x=district_df['Date'],
                        y=normalize(district_df['Total_Admissions']),
                        name="Total Admissions (normalized)",
                        mode='lines+markers',
                        line=dict(color="#3498db")
                    ))

                    # Optional features
                    for col, color in [
                        ('CCH_Air_temperature_ERA5_Land', '#e67e22'),
                        ('CCH_Precipitation_CHIRPS', '#2ecc71')
                    ]:
                        if col in district_df.columns:
                            fig_rel.add_trace(go.Scatter(
                                x=district_df['Date'],
                                y=normalize(district_df[col]),
                                name=f"{col} (normalized)",
                                mode='lines+markers',
                                line=dict(color=color)
                            ))

                    fig_rel.update_layout(
                        title=f"Normalized Admissions & Climate Features - {selected_district}",
                        xaxis_title="Date",
                        yaxis_title="Normalized Value",
                        height=500
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)

                with tab1:
                    st.subheader("Model Performance")
                    metrics = result['metrics']
                    st.write(metrics)
                    st.write("Accuracy (risk prediction):", result['accuracy'])
                    cm = pd.DataFrame(result['confusion_matrix'], index=result['risk_labels'], columns=result['risk_labels'])
                    st.markdown("### Confusion Matrix")
                    st.dataframe(cm, use_container_width=True)

                with tab2:
                    st.subheader("Forecasts")
                    st.dataframe(forecast_df, use_container_width=True)

                with tab3:
                    st.subheader("Geographic Risk Distribution")

                    if geo_file is not None:
                        try:
                            # Load GeoJSON
                            gdf = gpd.read_file(geo_file)

                            # Identify district column
                            name_col = next((c for c in gdf.columns if 'name' in c.lower() or 'dist' in c.lower()), None)
                            if name_col:
                                gdf = gdf.rename(columns={name_col: 'District'})
                            else:
                                st.warning("Could not find district name column in GeoJSON")
                                st.stop()

                            # Prepare forecast data
                            forecast_df_display = forecast_df.copy()
                            forecast_df_display['Month_Year'] = forecast_df_display['Date'].dt.strftime('%B %Y')

                            # Choose month to display
                            selected_map_month = st.selectbox("Select Month:", forecast_df_display['Month_Year'].unique())
                            month_data = forecast_df_display[forecast_df_display['Month_Year'] == selected_map_month]

                            # Merge GeoJSON with forecast
                            map_data = gdf.merge(month_data, on='District', how='left')

                            # Create folium map
                            center_lat = map_data.geometry.centroid.y.mean()
                            center_lon = map_data.geometry.centroid.x.mean()
                            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

                            # Color dictionary
                            RISK_COLORS = Config.RISK_COLORS

                            # Add polygons
                            for _, row in map_data.iterrows():
                                risk = row.get('Risk', 'No Data')
                                color = RISK_COLORS.get(risk, "#95a5a6")
                                geo_json = folium.GeoJson(
                                    row['geometry'],
                                    style_function=lambda feature, color=color: {
                                        'fillColor': color,
                                        'color': 'black',
                                        'weight': 1,
                                        'fillOpacity': 0.6
                                    },
                                    tooltip=folium.Tooltip(f"{row['District']}: {risk}")
                                )
                                geo_json.add_to(m)

                            # Display map in Streamlit
                            st_folium(m, width=700, height=500)

                        except Exception as e:
                            st.error(f"Error displaying map: {str(e)}")
                    else:
                        st.info("Upload GeoJSON file to view risk maps.")

                with tab4:
                    st.subheader("Feature Importance")
                    model = result['model']
                    if hasattr(model, 'feature_importances_'):
                        importances = pd.DataFrame({
                            'Feature': features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        st.dataframe(importances, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type.")

        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()