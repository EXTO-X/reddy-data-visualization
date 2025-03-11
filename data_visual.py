import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")
st.title("Interactive Data Analysis Dashboard")
st.markdown("Explore the synthetic dataset with various visualizations")

# Generate a synthetic dataset
@st.cache_data
def generate_data():
    np.random.seed(42)
    num_records = 100

    data = {
        "Date": pd.date_range(start="2024-01-01", periods=num_records, freq="D"),
        "Revenue": np.random.randint(1000, 10000, size=num_records),
        "Expenses": np.random.randint(500, 8000, size=num_records),
        "Customers": np.random.randint(10, 1000, size=num_records),
    }

    df = pd.DataFrame(data)
    # Convert Date to string immediately to avoid PyArrow issues
    df['Date'] = df['Date'].astype(str)

    # Introduce some missing values randomly
    df.loc[np.random.choice(df.index, size=5, replace=False), "Revenue"] = np.nan
    df.loc[np.random.choice(df.index, size=5, replace=False), "Expenses"] = np.nan
    
    return df

df = generate_data()

# 1. Data Preprocessing
st.header("Data Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data")
    # Convert Date to string format before displaying
    display_df = df.copy()
    display_df['Date'] = display_df['Date'].astype(str)
    st.write(display_df)  # Use st.write instead of st.dataframe

with col2:
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    st.write(missing_data)

# Handling Missing Values
fill_method = st.selectbox("Select method to handle missing values:", 
                          ["Mean", "Median", "Forward Fill", "Backward Fill", "None"])

if fill_method != "None":
    # Create a temporary dataframe without the Date column for calculations
    numeric_df = df.select_dtypes(include=[np.number])
    
    if fill_method == "Mean":
        # Calculate mean only for numeric columns
        means = numeric_df.mean()
        df_clean = df.copy()
        for col in numeric_df.columns:
            df_clean[col] = df_clean[col].fillna(means[col])
    elif fill_method == "Median":
        # Calculate median only for numeric columns
        medians = numeric_df.median()
        df_clean = df.copy()
        for col in numeric_df.columns:
            df_clean[col] = df_clean[col].fillna(medians[col])
    elif fill_method == "Forward Fill":
        df_clean = df.ffill()
    elif fill_method == "Backward Fill":
        df_clean = df.bfill()
else:
    df_clean = df.copy()

# 2. Statistical Summary
st.header("Statistical Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Summary Statistics")
    st.write(df_clean.describe())

with col2:
    st.subheader("Correlation Matrix")
    correlation_matrix = df_clean.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(correlation_matrix, 
                   text_auto=True, 
                   color_continuous_scale='RdBu_r',
                   title="Feature Correlation Heatmap")
    st.plotly_chart(fig)

# 3. Data Visualization
st.header("Data Visualization")

# Feature selection for visualization
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Distribution plots
st.subheader("Distribution of Features")
selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df_clean, x=selected_feature, 
                      title=f"Distribution of {selected_feature}",
                      marginal="box")
    st.plotly_chart(fig)

with col2:
    fig = px.box(df_clean, y=selected_feature, 
                title=f"Box Plot of {selected_feature}")
    st.plotly_chart(fig)

# Time Series Analysis
st.header("Time Series Analysis")

# Set date as index for time series analysis
df_ts = df_clean.copy()
# Convert string dates back to datetime for time series analysis
df_ts['Date'] = pd.to_datetime(df_ts['Date'])
df_ts.set_index("Date", inplace=True)

# Feature selection for time series
ts_feature = st.selectbox("Select feature for time series analysis:", numeric_cols)

# Moving average window
ma_window = st.slider("Moving Average Window Size:", min_value=1, max_value=30, value=5)

# Calculate moving average
df_ts[f"{ts_feature}_MA"] = df_ts[ts_feature].rolling(window=ma_window).mean()

# Plot time series with moving average
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts[ts_feature], 
                        mode='lines+markers', name=ts_feature))
fig.add_trace(go.Scatter(x=df_ts.index, y=df_ts[f"{ts_feature}_MA"], 
                        mode='lines', name=f"{ma_window}-Day Moving Average",
                        line=dict(color='red', width=2)))
fig.update_layout(title=f"{ts_feature} Over Time with Moving Average",
                 xaxis_title="Date", yaxis_title=ts_feature,
                 height=500)
st.plotly_chart(fig)

# 4. Feature Relationships
st.header("Feature Relationships")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox("Select X-axis feature:", numeric_cols, index=0)
    y_feature = st.selectbox("Select Y-axis feature:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    # Remove trendline to avoid statsmodels dependency
    fig = px.scatter(df_clean, x=x_feature, y=y_feature, 
                    title=f"{y_feature} vs {x_feature}")
    st.plotly_chart(fig)

with col2:
    # Calculate profit if revenue and expenses are selected
    if "Revenue" in df_clean.columns and "Expenses" in df_clean.columns:
        df_clean["Profit"] = df_clean["Revenue"] - df_clean["Expenses"]
        
        # Create a copy with string dates for plotting
        plot_df = df_clean.copy()
        plot_df['Date'] = plot_df['Date'].astype(str)
        
        fig = px.bar(plot_df, x="Date", y=["Revenue", "Expenses", "Profit"],
                    title="Revenue, Expenses and Profit",
                    barmode="group")
        st.plotly_chart(fig)
    else:
        st.info("Revenue and Expenses columns are required to calculate profit.")

st.markdown("---")
st.markdown("Dashboard created with Streamlit")