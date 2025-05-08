import time
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet
from prophet.serialize import model_from_json

start = time.time()

# --- Page Config ---
st.set_page_config(
    page_title="Renewable Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-title {
        color: #6c757d;
        font-size: 14px;
    }
    .metric-value {
        color: #212529;
        font-size: 24px;
        font-weight: bold;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .header {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    canvas { 
        -moz-box-shadow: none !important;
        -webkit-box-shadow: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Data ---
@st.cache_data
def load_data():
    url = "data/processed/Feature_Engineering _Dataset/feature_engineered_data.csv"
    cols = ['time', 'Country', 'Solar', 'Wind Onshore', 'temp', 'rhum', 'prcp', 'wspd', 'pres']
    df = pd.read_csv(url, usecols=cols)
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['month_name'] = df['time'].dt.month_name()
    df['season'] = df['time'].dt.month % 12 // 3 + 1
    return df


df = load_data()


# --- Load Prophet Models ---
@st.cache_resource
def load_prophet_models():
    # Load solar model
    with open('dashboard/model_data/solar_prophet_model.json', 'r') as fin:
        solar_model = model_from_json(fin.read())

    # Load wind model
    with open('dashboard/model_data/wind_prophet_model.json', 'r') as fin:
        wind_model = model_from_json(fin.read())

    # Get list of extra regressors from the models
    solar_regressors = list(solar_model.extra_regressors.keys())
    wind_regressors = list(wind_model.extra_regressors.keys())

    return solar_model, wind_model, solar_regressors, wind_regressors


solar_model, wind_model, solar_regressors, wind_regressors = load_prophet_models()

# --- Header ---
st.markdown("""
<div class="header">
    <h1 style="color:#2c3e50;">⚡ Renewable Energy Dashboard</h1>
    <p style="color:#34495e;">
        Track and forecast solar and wind energy production across France, Italy, and Spain
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
countries = st.sidebar.multiselect(
    "Select Countries",
    options=df['Country'].unique(),
    default=df['Country'].unique()
)

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['time'].dt.year.min()),
    max_value=int(df['time'].dt.year.max()),
    value=(int(df['time'].dt.year.min()), int(df['time'].dt.year.max()))
)

energy_types = st.sidebar.multiselect(
    "Energy Types",
    options=['Solar', 'Wind Onshore'],
    default=['Solar', 'Wind Onshore']
)

forecast_period = st.sidebar.selectbox(
    "Forecast Period",
    options=['24 hours', '1 week', '1 month'],
    index=0
)

# Apply filters
filtered_df = df[
    (df['Country'].isin(countries)) &
    (df['time'].dt.year.between(year_range[0], year_range[1]))
    ]

# --- Key Metrics ---
st.subheader("🌍 Regional Energy Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_solar = filtered_df['Solar'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Solar Production</div>
        <div class="metric-value">{total_solar:,.0f} units</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_wind = filtered_df['Wind Onshore'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Wind Production</div>
        <div class="metric-value">{total_wind:,.0f} units</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    solar_ratio = total_solar / (total_solar + total_wind) if (total_solar + total_wind) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Solar/Wind Ratio</div>
        <div class="metric-value">{solar_ratio:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    peak_solar = filtered_df['Solar'].max()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Peak Solar Output</div>
        <div class="metric-value">{peak_solar:,.0f} units</div>
    </div>
    """, unsafe_allow_html=True)

# --- Country Comparison ---
st.subheader("🏁 Country Comparison")
tab1, tab2, tab3 = st.tabs(["Production Trends", "Seasonal Patterns", "Hourly Patterns"])

with tab1:
    # Melt the dataframe to have energy types as a variable
    country_energy = filtered_df.groupby(['Country', filtered_df['time'].dt.year])[
        ['Solar', 'Wind Onshore']].sum().reset_index()
    melted_df = country_energy.melt(
        id_vars=['Country', 'time'],
        value_vars=['Solar', 'Wind Onshore'],
        var_name='Energy Type',
        value_name='Production'
    )

    # Create the plot
    fig = px.line(
        melted_df,
        x='time',
        y='Production',
        color='Country',
        line_dash='Energy Type',
        title='Annual Energy Production by Country and Type',
        labels={'time': 'Year', 'Production': 'Energy Production'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Customize the legend and lines
    fig.update_layout(
        legend_title_text='Country',
        legend_itemsizing='constant',  # Makes legend items same size
        hovermode='x unified'
    )

    # Explicitly set line styles for each energy type
    fig.update_traces(
        line=dict(width=3),
        selector=dict(line_dash='solid')  # Default for Solar
    )
    fig.update_traces(
        line=dict(width=3, dash='dot'),
        selector=dict(line_dash='dot')  # For Wind Onshore
    )

    # Improve hover template
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Year: %{x}<br>' +
                      'Production: %{y:,} units<extra></extra>'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Create season mapping and ordered categories
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    filtered_df['season_name'] = filtered_df['season'].map(season_map)
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']

    # Melt the data to combine energy types
    season_energy = filtered_df.groupby(['Country', 'season_name'])[['Solar', 'Wind Onshore']].mean().reset_index()
    melted_df = season_energy.melt(
        id_vars=['Country', 'season_name'],
        value_vars=['Solar', 'Wind Onshore'],
        var_name='Energy Type',
        value_name='Production'
    )

    # Create ordered category for seasons
    melted_df['season_name'] = pd.Categorical(
        melted_df['season_name'],
        categories=season_order,
        ordered=True
    )

    # Create stacked bar chart with seasons as x-axis
    fig = px.bar(
        melted_df,
        x='season_name',  # Seasons on x-axis
        y='Production',
        color='Energy Type',
        facet_col='Country',  # Separate subplots per country
        title='Average Seasonal Production Composition',
        category_orders={"season_name": season_order},
        color_discrete_map={
            'Solar': '#FFA500',  # Orange for solar
            'Wind Onshore': '#4682B4'  # Steel blue for wind
        }
    )

    # Improve layout
    fig.update_layout(
        barmode='stack',
        hovermode='x unified',
        yaxis_title='Average Production (units)',
        showlegend=True,
        xaxis_title='',  # Remove x-axis title
        xaxis2_title='',  # Remove x-axis title for subplots
        xaxis3_title=''  # Remove x-axis title for subplots
    )

    # Remove the "Country=" prefix from facet titles
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Season: %{x}<br>' +
                      'Production: %{y:.2f} units<extra></extra>'
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Melt the dataframe to have energy types as a variable
    hourly_energy = filtered_df.groupby(['Country', 'hour'])[['Solar', 'Wind Onshore']].mean().reset_index()
    melted_df = hourly_energy.melt(
        id_vars=['Country', 'hour'],
        value_vars=['Solar', 'Wind Onshore'],
        var_name='Energy Type',
        value_name='Production'
    )

    # Create the plot
    fig = px.line(
        melted_df,
        x='hour',
        y='Production',
        color='Country',
        line_dash='Energy Type',
        title='Average Daily Production Pattern',
        labels={'hour': 'Hour of Day', 'Production': 'Average Production'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Customize the legend and lines
    fig.update_layout(
        legend_title_text='Country',
        legend_itemsizing='constant',
        hovermode='x unified',
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            range=[0, 23]
        ),
        yaxis_title='Average Energy Production (units)'
    )

    # Explicitly set line styles
    fig.update_traces(
        line=dict(width=3),
        selector=dict(line_dash='solid')  # Solar gets solid lines
    )
    fig.update_traces(
        line=dict(width=3, dash='dot'),
        selector=dict(line_dash='dot')  # Wind gets dotted lines
    )

    # Improve hover template
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>' +
                     'Hour: %{x}:00<br>' +
                     'Production: %{y:.2f} units<extra></extra>'
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Prophet Forecasting ---
st.subheader("🔮 Energy Production Forecast")

if 'Solar' in energy_types or 'Wind Onshore' in energy_types:
    with st.spinner('Preparing forecast...'):
        # Determine forecast period
        if forecast_period == '24 hours':
            periods = 24
            freq = 'h'
        elif forecast_period == '1 week':
            periods = 168  # 24*7
            freq = 'h'
        else:  # 1 month
            periods = 30
            freq = 'D'

        # Make future dataframe
        last_date = filtered_df['time'].max()
        future = pd.DataFrame({'ds': pd.date_range(
            start=last_date,
            periods=periods + 1,
            freq=freq
        )})

        # Add required regressors with default values (0)
        if 'Solar' in energy_types:
            for reg in solar_regressors:
                future[reg] = 0

        if 'Wind Onshore' in energy_types:
            for reg in wind_regressors:
                future[reg] = 0

        # For Solar Forecast
        if 'Solar' in energy_types:
            st.write("### Solar Energy Forecast")
            try:
                solar_forecast = solar_model.predict(future)

                # Show forecast components
                st.write("#### Forecast Components")
                fig_solar_components = solar_model.plot_components(solar_forecast)

                # Customize the trend plot
                ax = fig_solar_components.axes[0]
                ax.set_title("Long-Term Solar Production Trend", pad=20)
                ax.set_xlabel("Date")
                ax.set_ylabel("Solar Production (units)")

                # Remove the fifth plot if it exists
                if len(fig_solar_components.axes) >= 5:
                    fig_solar_components.delaxes(fig_solar_components.axes[4])

                st.pyplot(fig_solar_components)

                # Show forecast plot
                st.write("#### Forecast Visualization")
                fig_solar_forecast = solar_model.plot(solar_forecast)
                st.pyplot(fig_solar_forecast)

                # Show latest prediction
                last_solar_pred = solar_forecast.iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Solar Output</div>
                    <div class="metric-value">{last_solar_pred['yhat']:.2f} units</div>
                    <div>at {last_solar_pred['ds'].strftime('%Y-%m-%d %H:%M')}</div>
                    <div class="metric-title">Uncertainty Range: {last_solar_pred['yhat_lower']:.2f} to {last_solar_pred['yhat_upper']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in solar forecast: {str(e)}")

        # Wind Forecast
        if 'Wind Onshore' in energy_types:
            st.write("### Wind Energy Forecast")
            try:
                wind_forecast = wind_model.predict(future)

                # Show forecast components
                st.write("#### Forecast Components")
                fig_wind_components = wind_model.plot_components(wind_forecast)
                st.pyplot(fig_wind_components)

                # Show forecast plot
                st.write("#### Forecast Visualization")
                fig_wind_forecast = wind_model.plot(wind_forecast)
                st.pyplot(fig_wind_forecast)

                # Show latest prediction
                last_wind_pred = wind_forecast.iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Wind Output</div>
                    <div class="metric-value">{last_wind_pred['yhat']:.2f} units</div>
                    <div>at {last_wind_pred['ds'].strftime('%Y-%m-%d %H:%M')}</div>
                    <div class="metric-title">Uncertainty Range: {last_wind_pred['yhat_lower']:.2f} to {last_wind_pred['yhat_upper']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in wind forecast: {str(e)}")

# --- Data Export ---
st.subheader("📊 Data Export")
if st.button("Generate Custom Report"):
    report_cols = ['time', 'Country', 'Solar', 'Wind Onshore', 'temp', 'rhum', 'prcp', 'wspd', 'pres']
    report_df = filtered_df[report_cols].copy()
    csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Current View as CSV",
        data=csv,
        file_name='renewable_energy_report.csv',
        mime='text/csv'
    )

# --- Footer ---
st.markdown("---")
st.markdown("""
**Data Sources**: 
- Historical weather data from [OpenWeather]
- Energy production data from European transmission system operators

**Methodology**:
- Facebook Prophet model for time series forecasting
- Accounts for daily, weekly, and yearly seasonality
- Includes uncertainty intervals in predictions
""")

st.write(f"Execution time: {time.time() - start:.2f}s")