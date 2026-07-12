from datetime import date
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"

FORECAST_DATA_PATH = DATA_DIR / "forecast_features.csv"


# =========================================================
# CONSTANTS
# =========================================================

FEATURE_COLUMNS = [
    "Year",
    "Month",
    "Quarter",
    "Week",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "IsWeekend",
    "Lag_1",
    "Lag_7",
    "Lag_30",
    "Rolling_Mean_7",
    "Rolling_Mean_30",
]

MODEL_PATHS = {
    "Linear Regression": MODEL_DIR / "linear_regression.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
    "XGBoost": MODEL_DIR / "xgboost.pkl",
}

MODEL_METRICS = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
        ],
        "MAE": [
            1358.46,
            1367.22,
            1405.12,
        ],
        "RMSE": [
            2313.59,
            2451.95,
            2525.25,
        ],
        "R²": [
            0.2392,
            0.1455,
            0.0937,
        ],
    }
)

WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


# =========================================================
# DATA AND MODEL LOADING
# =========================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and validate the processed forecasting dataset."""

    if not FORECAST_DATA_PATH.exists():
        raise FileNotFoundError(
            "The forecasting dataset could not be found at:\n"
            f"{FORECAST_DATA_PATH}"
        )

    df = pd.read_csv(FORECAST_DATA_PATH)

    required_columns = [
        "Order Date",
        "Sales",
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The forecasting dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce",
    )

    numeric_columns = [
        "Sales",
        *FEATURE_COLUMNS,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "Order Date",
                "Sales",
                *FEATURE_COLUMNS,
            ]
        )
        .sort_values("Order Date")
        .reset_index(drop=True)
    )

    return df


@st.cache_resource
def load_forecasting_models() -> tuple[dict, dict]:
    """Load trained forecasting models."""

    loaded_models = {}
    model_errors = {}

    for model_name, model_path in MODEL_PATHS.items():

        if not model_path.exists():
            model_errors[model_name] = (
                f"File not found: {model_path.name}"
            )
            continue

        try:
            with open(model_path, "rb") as model_file:
                loaded_models[model_name] = pickle.load(model_file)

        except Exception as error:
            model_errors[model_name] = str(error)

    return loaded_models, model_errors


try:
    forecast_df = load_data()

except Exception as error:
    st.error(
        "The dashboard could not load the forecasting dataset."
    )
    st.exception(error)
    st.stop()


forecasting_models, model_loading_errors = (
    load_forecasting_models()
)


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_currency(value: float) -> str:
    """Format a number as US currency."""

    return f"${float(value):,.2f}"


def build_feature_row(
    forecast_date: date,
    lag_1: float,
    lag_7: float,
    lag_30: float,
    rolling_mean_7: float,
    rolling_mean_30: float,
) -> pd.DataFrame:
    """Create a model-ready forecasting feature row."""

    timestamp = pd.Timestamp(forecast_date)

    year = timestamp.year
    month = timestamp.month
    quarter = timestamp.quarter
    week = int(timestamp.isocalendar().week)
    day = timestamp.day
    day_of_week = timestamp.dayofweek
    day_of_year = timestamp.dayofyear
    is_weekend = int(day_of_week >= 5)

    input_data = pd.DataFrame(
        [
            {
                "Year": year,
                "Month": month,
                "Quarter": quarter,
                "Week": week,
                "Day": day,
                "DayOfWeek": day_of_week,
                "DayOfYear": day_of_year,
                "IsWeekend": is_weekend,
                "Lag_1": float(lag_1),
                "Lag_7": float(lag_7),
                "Lag_30": float(lag_30),
                "Rolling_Mean_7": float(rolling_mean_7),
                "Rolling_Mean_30": float(rolling_mean_30),
            }
        ]
    )

    return input_data[FEATURE_COLUMNS]


def align_features_with_model(
    model,
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    """Align dataframe columns with the saved model."""

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:
            raise ValueError(
                "The model expects unavailable features: "
                + ", ".join(missing_features)
            )

        return input_data[expected_features]

    return input_data[FEATURE_COLUMNS]


def calculate_input_reliability(
    input_data: pd.DataFrame,
    training_data: pd.DataFrame,
) -> tuple[str, str]:
    """Check whether historical inputs fall within training ranges."""

    historical_features = [
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30",
    ]

    outside_range_features = []

    for feature in historical_features:
        value = float(input_data.iloc[0][feature])
        training_min = float(training_data[feature].min())
        training_max = float(training_data[feature].max())

        if value < training_min or value > training_max:
            outside_range_features.append(feature)

    outside_range_count = len(outside_range_features)

    if outside_range_count == 0:
        return (
            "High input reliability",
            "All historical inputs are within the ranges observed "
            "in the training dataset.",
        )

    if outside_range_count <= 2:
        return (
            "Moderate input reliability",
            f"{outside_range_count} historical input value(s) are "
            "outside the training ranges: "
            + ", ".join(outside_range_features),
        )

    return (
        "Low input reliability",
        f"{outside_range_count} historical inputs are outside the "
        "training ranges: "
        + ", ".join(outside_range_features),
    )


def create_forecast_comparison_chart(
    predicted_sales: float,
    historical_average: float,
    rolling_average: float,
) -> go.Figure:
    """Create contextual forecast comparison chart."""

    chart_data = pd.DataFrame(
        {
            "Value Type": [
                "Predicted Sales",
                "Historical Average",
                "Entered 30-Observation Mean",
            ],
            "Sales": [
                predicted_sales,
                historical_average,
                rolling_average,
            ],
        }
    )

    figure = px.bar(
        chart_data,
        x="Value Type",
        y="Sales",
        text="Sales",
        title="Forecast Context Comparison",
    )

    figure.update_traces(
        texttemplate="$%{y:,.2f}",
        textposition="outside",
    )

    figure.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="",
        yaxis_title="Sales",
        margin=dict(t=80),
    )

    return figure


def create_model_metrics_chart(
    metric: str,
) -> go.Figure:
    """Create a model evaluation comparison chart."""

    ascending = metric != "R²"

    chart_data = MODEL_METRICS.sort_values(
        metric,
        ascending=ascending,
    )

    figure = px.bar(
        chart_data,
        x="Model",
        y=metric,
        title=f"Model Comparison by {metric}",
    )

    if metric == "R²":
        figure.update_traces(
            text=chart_data[metric].map(
                lambda value: f"{value:.4f}"
            ),
            textposition="outside",
        )

        figure.update_layout(
            yaxis_tickformat=".2f",
        )

    else:
        figure.update_traces(
            text=chart_data[metric].map(
                lambda value: f"{value:,.2f}"
            ),
            textposition="outside",
        )

    figure.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="",
        yaxis_title=metric,
        margin=dict(t=80),
    )

    return figure


def get_best_model_name() -> str:
    """Return the model with the highest R²."""

    best_row = MODEL_METRICS.sort_values(
        "R²",
        ascending=False,
    ).iloc[0]

    return str(best_row["Model"])


def get_model_feature_importance(
    model_name: str,
    model,
) -> pd.DataFrame | None:
    """Extract feature coefficients or importance values."""

    if model_name == "Linear Regression":

        if not hasattr(model, "coef_"):
            return None

        coefficients = np.asarray(model.coef_).flatten()

        if len(coefficients) != len(FEATURE_COLUMNS):
            return None

        importance_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Raw Value": coefficients,
                "Importance": np.abs(coefficients),
                "Direction": np.where(
                    coefficients >= 0,
                    "Positive",
                    "Negative",
                ),
            }
        )

    else:

        if not hasattr(model, "feature_importances_"):
            return None

        importances = np.asarray(
            model.feature_importances_
        ).flatten()

        if len(importances) != len(FEATURE_COLUMNS):
            return None

        importance_df = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Raw Value": importances,
                "Importance": importances,
                "Direction": "Relative importance",
            }
        )

    return (
        importance_df.sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_normalized_explainability_table(
    models: dict,
) -> pd.DataFrame:
    """Create normalized importance table across models."""

    comparison_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
        }
    )

    for model_name in MODEL_PATHS:

        model = models.get(model_name)

        if model is None:
            comparison_df[model_name] = np.nan
            continue

        importance_df = get_model_feature_importance(
            model_name,
            model,
        )

        if importance_df is None:
            comparison_df[model_name] = np.nan
            continue

        importance_map = importance_df.set_index(
            "Feature"
        )["Importance"]

        values = comparison_df["Feature"].map(
            importance_map
        ).fillna(0.0)

        maximum_value = float(values.max())

        if maximum_value > 0:
            values = values / maximum_value

        comparison_df[model_name] = values

    available_columns = [
        model_name
        for model_name in MODEL_PATHS
        if comparison_df[model_name].notna().any()
    ]

    if available_columns:
        comparison_df["Average Importance"] = (
            comparison_df[available_columns]
            .mean(axis=1)
        )

        comparison_df = comparison_df.sort_values(
            "Average Importance",
            ascending=False,
        )

    return comparison_df.reset_index(drop=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📈 Sales Forecasting")

st.sidebar.caption(
    "Demand Intelligence Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Dataset Overview",
        "📈 Sales Trends",
        "🔮 Forecast",
        "⚖️ Model Comparison",
        "🧠 Model Explainability",
        "ℹ️ About Project",
    ],
)

st.sidebar.divider()

available_model_count = len(forecasting_models)

st.sidebar.metric(
    "Available Models",
    f"{available_model_count}/{len(MODEL_PATHS)}",
)

st.sidebar.metric(
    "Dataset Records",
    f"{len(forecast_df):,}",
)

st.sidebar.metric(
    "Engineered Features",
    len(FEATURE_COLUMNS),
)

if model_loading_errors:
    with st.sidebar.expander(
        "Model loading information"
    ):

        for model_name, error_message in (
            model_loading_errors.items()
        ):
            st.warning(
                f"**{model_name}:** {error_message}"
            )

st.sidebar.divider()

st.sidebar.caption(
    "End-to-end machine learning portfolio project"
)


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.title("📈 Sales Forecasting Dashboard")

st.markdown(
    """
    Machine learning-based retail sales forecasting using
    **Linear Regression, Random Forest, and XGBoost**.
    """
)

st.divider()


# =========================================================
# HOME DASHBOARD PAGE
# =========================================================

if page == "🏠 Dashboard":

    best_model_name = get_best_model_name()

    best_model_row = MODEL_METRICS.loc[
        MODEL_METRICS["Model"] == best_model_name
    ].iloc[0]

    st.header("🏠 Executive Dashboard")

    st.markdown(
        """
        Monitor historical sales performance, model quality, and
        forecasting-system readiness from one consolidated view.
        """
    )

    dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = (
        st.columns(4)
    )

    dashboard_col1.metric(
        "Historical Records",
        f"{len(forecast_df):,}",
    )

    dashboard_col2.metric(
        "Total Sales",
        format_currency(forecast_df["Sales"].sum()),
    )

    dashboard_col3.metric(
        "Selected Model",
        best_model_name,
    )

    dashboard_col4.metric(
        "Best R²",
        f"{best_model_row['R²']:.4f}",
    )

    dashboard_row2_col1, dashboard_row2_col2, dashboard_row2_col3 = (
        st.columns(3)
    )

    dashboard_row2_col1.metric(
        "Average Daily Sales",
        format_currency(forecast_df["Sales"].mean()),
    )

    dashboard_row2_col2.metric(
        "Median Daily Sales",
        format_currency(forecast_df["Sales"].median()),
    )

    dashboard_row2_col3.metric(
        "Maximum Daily Sales",
        format_currency(forecast_df["Sales"].max()),
    )

    st.subheader("Historical Sales Performance")

    dashboard_monthly_sales = (
        forecast_df.assign(
            Period=forecast_df["Order Date"].dt.to_period("M")
        )
        .groupby("Period", as_index=False)["Sales"]
        .sum()
    )

    dashboard_monthly_sales["Period"] = (
        dashboard_monthly_sales["Period"].dt.to_timestamp()
    )

    dashboard_sales_figure = px.line(
        dashboard_monthly_sales,
        x="Period",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend",
    )

    dashboard_sales_figure.update_layout(
        template="plotly_dark",
        height=460,
        xaxis_title="Month",
        yaxis_title="Sales",
    )

    st.plotly_chart(
        dashboard_sales_figure,
        width="stretch",
        key="dashboard_monthly_sales_chart",
    )

    dashboard_chart_col1, dashboard_chart_col2 = (
        st.columns(2)
    )

    with dashboard_chart_col1:

        st.subheader("Model Performance")

        dashboard_metrics_figure = (
            create_model_metrics_chart("MAE")
        )

        st.plotly_chart(
            dashboard_metrics_figure,
            width="stretch",
            key="dashboard_model_mae_chart",
        )

    with dashboard_chart_col2:

        st.subheader("System Status")

        st.success(
            f"""
            **Production candidate:** {best_model_name}

            The selected model achieved the lowest MAE and RMSE
            and the highest R² among the evaluated models.
            """
        )

        st.markdown(
            """
            **Available capabilities**

            - Interactive single-date forecasting
            - Automatic calendar-feature generation
            - Lag and rolling-average inputs
            - Input-range reliability checking
            - Prediction-history downloads
            - Interactive trend analysis
            - Cross-model explainability
            """
        )

    st.subheader("Business Insights")

    highest_sales_row = forecast_df.loc[
        forecast_df["Sales"].idxmax()
    ]

    monthly_average = (
        forecast_df.groupby("Month")["Sales"]
        .mean()
    )

    strongest_month_number = int(
        monthly_average.idxmax()
    )

    quarter_totals = (
        forecast_df.groupby("Quarter")["Sales"]
        .sum()
    )

    strongest_quarter = int(
        quarter_totals.idxmax()
    )

    weekday_average = (
        forecast_df.groupby("DayOfWeek")["Sales"]
        .mean()
    )

    strongest_weekday_number = int(
        weekday_average.idxmax()
    )

    insight_col1, insight_col2, insight_col3 = (
        st.columns(3)
    )

    insight_col1.info(
        f"""
        **Highest recorded sales**

        {format_currency(highest_sales_row["Sales"])}

        Date: {highest_sales_row["Order Date"].date()}
        """
    )

    insight_col2.info(
        f"""
        **Strongest average month**

        {MONTH_NAMES[strongest_month_number]}

        Average: {
            format_currency(
                monthly_average.loc[strongest_month_number]
            )
        }
        """
    )

    insight_col3.info(
        f"""
        **Strongest weekday**

        {WEEKDAY_NAMES[strongest_weekday_number]}

        Highest-sales quarter: Q{strongest_quarter}
        """
    )


# =========================================================
# DATASET OVERVIEW PAGE
# =========================================================

elif page == "📊 Dataset Overview":

    st.header("📊 Dataset Overview")

    st.markdown(
        """
        Review the processed forecasting dataset used for feature
        engineering, model training, and evaluation.
        """
    )

    total_records = len(forecast_df)
    total_features = forecast_df.shape[1]

    start_date = forecast_df["Order Date"].min().date()
    end_date = forecast_df["Order Date"].max().date()

    average_sales = forecast_df["Sales"].mean()
    median_sales = forecast_df["Sales"].median()
    maximum_sales = forecast_df["Sales"].max()

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    row1_col1.metric(
        "Total Records",
        f"{total_records:,}",
    )

    row1_col2.metric(
        "Dataset Columns",
        total_features,
    )

    row1_col3.metric(
        "Date Range",
        f"{start_date} → {end_date}",
    )

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    row2_col1.metric(
        "Average Sales",
        format_currency(average_sales),
    )

    row2_col2.metric(
        "Median Sales",
        format_currency(median_sales),
    )

    row2_col3.metric(
        "Maximum Sales",
        format_currency(maximum_sales),
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        forecast_df.head(15),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Feature Summary")

    feature_summary = (
        forecast_df.select_dtypes(include="number")
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    st.dataframe(
        feature_summary,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Missing-Value Audit")

    missing_summary = pd.DataFrame(
        {
            "Feature": forecast_df.columns,
            "Missing Values": (
                forecast_df.isna().sum().values
            ),
            "Missing Percentage": (
                forecast_df.isna().mean().values * 100
            ),
        }
    )

    st.dataframe(
        missing_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "Missing Percentage": (
                st.column_config.NumberColumn(
                    "Missing Percentage",
                    format="%.2f%%",
                )
            ),
        },
    )

    csv_data = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Forecasting Dataset",
        data=csv_data,
        file_name="forecast_features.csv",
        mime="text/csv",
        width="stretch",
    )


# =========================================================
# SALES TRENDS PAGE
# =========================================================

elif page == "📈 Sales Trends":

    st.header("📈 Sales Trend Analysis")

    st.markdown(
        """
        Explore daily, monthly, yearly, weekday, and quarterly sales
        behaviour using interactive visualisations.
        """
    )

    minimum_date = forecast_df["Order Date"].min().date()
    maximum_date = forecast_df["Order Date"].max().date()

    selected_date_range = st.date_input(
        "Filter date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )

    filtered_df = forecast_df.copy()

    if (
        isinstance(selected_date_range, tuple)
        and len(selected_date_range) == 2
    ):
        selected_start, selected_end = selected_date_range

        filtered_df = forecast_df[
            forecast_df["Order Date"].dt.date.between(
                selected_start,
                selected_end,
            )
        ].copy()

    if filtered_df.empty:
        st.warning(
            "No records are available for the selected date range."
        )
        st.stop()

    trend_col1, trend_col2, trend_col3 = st.columns(3)

    trend_col1.metric(
        "Filtered Records",
        f"{len(filtered_df):,}",
    )

    trend_col2.metric(
        "Filtered Sales",
        format_currency(filtered_df["Sales"].sum()),
    )

    trend_col3.metric(
        "Average Daily Sales",
        format_currency(filtered_df["Sales"].mean()),
    )

    st.subheader("Daily Sales")

    daily_sales = (
        filtered_df.groupby(
            "Order Date",
            as_index=False,
        )["Sales"]
        .sum()
    )

    fig_daily = px.line(
        daily_sales,
        x="Order Date",
        y="Sales",
        title="Daily Sales Trend",
    )

    fig_daily.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Order Date",
        yaxis_title="Sales",
    )

    st.plotly_chart(
        fig_daily,
        width="stretch",
        key="sales_trends_daily_chart",
    )

    st.subheader("Monthly Sales")

    monthly_sales = (
        filtered_df.assign(
            Period=filtered_df["Order Date"].dt.to_period("M")
        )
        .groupby("Period", as_index=False)["Sales"]
        .sum()
    )

    monthly_sales["Period"] = (
        monthly_sales["Period"].dt.to_timestamp()
    )

    fig_monthly = px.bar(
        monthly_sales,
        x="Period",
        y="Sales",
        title="Monthly Sales",
    )

    fig_monthly.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Month",
        yaxis_title="Sales",
    )

    st.plotly_chart(
        fig_monthly,
        width="stretch",
        key="sales_trends_monthly_chart",
    )

    st.subheader("Yearly Sales")

    yearly_sales = (
        filtered_df.groupby(
            "Year",
            as_index=False,
        )["Sales"]
        .sum()
    )

    yearly_sales["Year"] = (
        yearly_sales["Year"].astype(int).astype(str)
    )

    fig_yearly = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        title="Yearly Sales",
    )

    fig_yearly.update_traces(
        text=yearly_sales["Sales"].map(
            lambda value: f"${value:,.0f}"
        ),
        textposition="inside",
    )

    fig_yearly.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Year",
        yaxis_title="Sales",
    )

    st.plotly_chart(
        fig_yearly,
        width="stretch",
        key="sales_trends_yearly_chart",
    )

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:

        weekday_sales = (
            filtered_df.groupby(
                "DayOfWeek",
                as_index=False,
            )["Sales"]
            .mean()
        )

        weekday_sales["Weekday"] = (
            weekday_sales["DayOfWeek"].map(
                WEEKDAY_NAMES
            )
        )

        fig_weekday = px.bar(
            weekday_sales,
            x="Weekday",
            y="Sales",
            title="Average Sales by Weekday",
        )

        fig_weekday.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Weekday",
            yaxis_title="Average Sales",
        )

        st.plotly_chart(
            fig_weekday,
            width="stretch",
            key="sales_trends_weekday_chart",
        )

    with analysis_col2:

        quarter_sales = (
            filtered_df.groupby(
                "Quarter",
                as_index=False,
            )["Sales"]
            .sum()
        )

        quarter_sales["Quarter"] = (
            "Q"
            + quarter_sales["Quarter"]
            .astype(int)
            .astype(str)
        )

        fig_quarter = px.bar(
            quarter_sales,
            x="Quarter",
            y="Sales",
            title="Sales by Quarter",
        )

        fig_quarter.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Quarter",
            yaxis_title="Sales",
        )

        st.plotly_chart(
            fig_quarter,
            width="stretch",
            key="sales_trends_quarter_chart",
        )

    st.subheader("Sales Distribution")

    fig_distribution = px.histogram(
        filtered_df,
        x="Sales",
        nbins=40,
        title="Distribution of Daily Sales",
    )

    fig_distribution.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Sales",
        yaxis_title="Frequency",
    )

    st.plotly_chart(
        fig_distribution,
        width="stretch",
        key="sales_distribution_chart",
    )


# =========================================================
# FORECAST PAGE
# =========================================================

elif page == "🔮 Forecast":

    st.header("🔮 Interactive Sales Forecast")

    st.markdown(
        """
        Select a forecasting model, choose a future date, and provide
        recent historical sales information to generate an estimated
        sales value.
        """
    )

    if not forecasting_models:
        st.error(
            "No trained forecasting models are available. Ensure the "
            "saved `.pkl` files exist inside the models folder."
        )
        st.stop()

    st.info(
        "Linear Regression produced the best evaluation results "
        "among the three trained models."
    )

    available_model_names = list(
        forecasting_models.keys()
    )

    default_model_index = (
        available_model_names.index(
            "Linear Regression"
        )
        if "Linear Regression" in available_model_names
        else 0
    )

    with st.form("forecast_form"):

        model_name = st.selectbox(
            "Select Forecasting Model",
            options=available_model_names,
            index=default_model_index,
        )

        st.subheader("Forecast Date")

        forecast_date = st.date_input(
            "Select the date to forecast",
            value=date(2019, 1, 1),
            min_value=date(2015, 1, 1),
            max_value=date(2030, 12, 31),
        )

        selected_timestamp = pd.Timestamp(
            forecast_date
        )

        derived_calendar_data = pd.DataFrame(
            {
                "Calendar Feature": [
                    "Year",
                    "Month",
                    "Quarter",
                    "Week of Year",
                    "Day of Month",
                    "Day of Week",
                    "Day of Year",
                    "Weekend",
                ],
                "Value": [
                    str(selected_timestamp.year),
                    str(selected_timestamp.month),
                    str(selected_timestamp.quarter),
                    str(
                        int(
                            selected_timestamp
                            .isocalendar()
                            .week
                        )
                    ),
                    str(selected_timestamp.day),
                    str(selected_timestamp.dayofweek),
                    str(selected_timestamp.dayofyear),
                    (
                        "Yes"
                        if selected_timestamp.dayofweek >= 5
                        else "No"
                    ),
                ],
            }
        )

        with st.expander(
            "View automatically generated calendar features"
        ):
            st.dataframe(
                derived_calendar_data,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Historical Sales Features")

        st.caption(
            "Default values are based on median values in the "
            "training dataset."
        )

        input_col1, input_col2 = st.columns(2)

        with input_col1:

            lag_1 = st.number_input(
                "Previous Sales Value — Lag 1",
                min_value=0.0,
                value=float(
                    forecast_df["Lag_1"].median()
                ),
                step=100.0,
                format="%.2f",
                help=(
                    "Sales recorded one observation before "
                    "the forecast."
                ),
            )

            lag_7 = st.number_input(
                "Sales 7 Observations Ago — Lag 7",
                min_value=0.0,
                value=float(
                    forecast_df["Lag_7"].median()
                ),
                step=100.0,
                format="%.2f",
                help=(
                    "Sales recorded seven observations "
                    "before the forecast."
                ),
            )

            lag_30 = st.number_input(
                "Sales 30 Observations Ago — Lag 30",
                min_value=0.0,
                value=float(
                    forecast_df["Lag_30"].median()
                ),
                step=100.0,
                format="%.2f",
                help=(
                    "Sales recorded thirty observations "
                    "before the forecast."
                ),
            )

        with input_col2:

            rolling_mean_7 = st.number_input(
                "7-Observation Rolling Mean",
                min_value=0.0,
                value=float(
                    forecast_df[
                        "Rolling_Mean_7"
                    ].median()
                ),
                step=100.0,
                format="%.2f",
                help=(
                    "Average sales across the previous "
                    "seven observations."
                ),
            )

            rolling_mean_30 = st.number_input(
                "30-Observation Rolling Mean",
                min_value=0.0,
                value=float(
                    forecast_df[
                        "Rolling_Mean_30"
                    ].median()
                ),
                step=100.0,
                format="%.2f",
                help=(
                    "Average sales across the previous "
                    "thirty observations."
                ),
            )

        submitted = st.form_submit_button(
            "Generate Forecast",
            type="primary",
            width="stretch",
        )

    if submitted:

        selected_model = forecasting_models.get(
            model_name
        )

        if selected_model is None:
            st.error(
                f"The saved {model_name} model could not be loaded."
            )

        else:

            try:
                input_data = build_feature_row(
                    forecast_date=forecast_date,
                    lag_1=lag_1,
                    lag_7=lag_7,
                    lag_30=lag_30,
                    rolling_mean_7=rolling_mean_7,
                    rolling_mean_30=rolling_mean_30,
                )

                model_input = align_features_with_model(
                    model=selected_model,
                    input_data=input_data,
                )

                raw_prediction = (
                    selected_model.predict(
                        model_input
                    )[0]
                )

                predicted_sales = max(
                    0.0,
                    float(raw_prediction),
                )

                reliability_label, reliability_message = (
                    calculate_input_reliability(
                        input_data=input_data,
                        training_data=forecast_df,
                    )
                )

                prediction_record = {
                    "Forecast Date": str(forecast_date),
                    "Model": model_name,
                    "Predicted Sales": predicted_sales,
                    "Input Reliability": reliability_label,
                    "Reliability Message": (
                        reliability_message
                    ),
                    **input_data.iloc[0].to_dict(),
                }

                st.session_state.latest_prediction = (
                    prediction_record
                )

                st.session_state.prediction_history.append(
                    prediction_record.copy()
                )

                st.success(
                    "Forecast generated successfully."
                )

            except Exception as error:
                st.error(
                    "The forecast could not be generated."
                )
                st.exception(error)

    if st.session_state.latest_prediction is not None:

        result = st.session_state.latest_prediction

        st.divider()
        st.subheader("Forecast Result")

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        result_col1.metric(
            "Predicted Sales",
            format_currency(
                result["Predicted Sales"]
            ),
        )

        result_col2.metric(
            "Selected Model",
            result["Model"],
        )

        result_col3.metric(
            "Forecast Date",
            result["Forecast Date"],
        )

        reliability = result["Input Reliability"]
        reliability_message = result.get(
            "Reliability Message",
            "",
        )

        if reliability == "High input reliability":
            st.success(
                f"{reliability}: {reliability_message}"
            )

        elif reliability == "Moderate input reliability":
            st.warning(
                f"{reliability}: {reliability_message}"
            )

        else:
            st.error(
                f"{reliability}: {reliability_message}"
            )

        comparison_figure = (
            create_forecast_comparison_chart(
                predicted_sales=result[
                    "Predicted Sales"
                ],
                historical_average=float(
                    forecast_df["Sales"].mean()
                ),
                rolling_average=float(
                    result["Rolling_Mean_30"]
                ),
            )
        )

        st.plotly_chart(
            comparison_figure,
            width="stretch",
            key="forecast_comparison_chart",
        )

        difference_from_average = (
            result["Predicted Sales"]
            - forecast_df["Sales"].mean()
        )

        percentage_difference = (
            difference_from_average
            / forecast_df["Sales"].mean()
            * 100
        )

        context_col1, context_col2 = st.columns(2)

        context_col1.metric(
            "Difference from Historical Average",
            format_currency(difference_from_average),
        )

        context_col2.metric(
            "Percentage Difference",
            f"{percentage_difference:+.2f}%",
        )

        with st.expander(
            "View complete model input"
        ):

            displayed_input = pd.DataFrame(
                [
                    {
                        column: result[column]
                        for column in FEATURE_COLUMNS
                    }
                ]
            )

            st.dataframe(
                displayed_input,
                width="stretch",
                hide_index=True,
            )

    if st.session_state.prediction_history:

        st.divider()
        st.subheader("Prediction History")

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        history_display_columns = [
            "Forecast Date",
            "Model",
            "Predicted Sales",
            "Input Reliability",
            "Lag_1",
            "Lag_7",
            "Lag_30",
            "Rolling_Mean_7",
            "Rolling_Mean_30",
        ]

        available_history_columns = [
            column
            for column in history_display_columns
            if column in history_df.columns
        ]

        st.dataframe(
            history_df[available_history_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "Predicted Sales": (
                    st.column_config.NumberColumn(
                        "Predicted Sales",
                        format="$%.2f",
                    )
                ),
            },
        )

        download_col1, download_col2 = (
            st.columns(2)
        )

        with download_col1:

            prediction_csv = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Prediction History",
                data=prediction_csv,
                file_name="sales_forecast_history.csv",
                mime="text/csv",
                width="stretch",
            )

        with download_col2:

            if st.button(
                "Clear Prediction History",
                width="stretch",
            ):
                st.session_state.prediction_history = []
                st.session_state.latest_prediction = None
                st.rerun()


# =========================================================
# MODEL COMPARISON PAGE
# =========================================================

elif page == "⚖️ Model Comparison":

    st.header("⚖️ Model Comparison")

    st.markdown(
        """
        Compare forecasting models using the evaluation results
        obtained during model training.

        - **MAE:** Lower values indicate smaller average errors.
        - **RMSE:** Lower values indicate better handling of large errors.
        - **R²:** Higher values indicate stronger explanatory performance.
        """
    )

    best_model_name = get_best_model_name()

    best_model_row = MODEL_METRICS.loc[
        MODEL_METRICS["Model"] == best_model_name
    ].iloc[0]

    best_col1, best_col2, best_col3, best_col4 = (
        st.columns(4)
    )

    best_col1.metric(
        "Selected Best Model",
        best_model_name,
    )

    best_col2.metric(
        "Best Model MAE",
        f"{best_model_row['MAE']:,.2f}",
    )

    best_col3.metric(
        "Best Model RMSE",
        f"{best_model_row['RMSE']:,.2f}",
    )

    best_col4.metric(
        "Best Model R²",
        f"{best_model_row['R²']:.4f}",
    )

    st.subheader("Evaluation Table")

    comparison_table = MODEL_METRICS.copy()

    comparison_table["Status"] = (
        comparison_table["Model"].apply(
            lambda model: (
                "Selected best model"
                if model == best_model_name
                else "Alternative model"
            )
        )
    )

    comparison_table["Model File"] = (
        comparison_table["Model"].apply(
            lambda model: (
                "Available"
                if model in forecasting_models
                else "Unavailable"
            )
        )
    )

    st.dataframe(
        comparison_table,
        width="stretch",
        hide_index=True,
        column_config={
            "MAE": st.column_config.NumberColumn(
                "MAE",
                format="%.2f",
            ),
            "RMSE": st.column_config.NumberColumn(
                "RMSE",
                format="%.2f",
            ),
            "R²": st.column_config.NumberColumn(
                "R²",
                format="%.4f",
            ),
        },
    )

    selected_metric = st.selectbox(
        "Choose a metric to visualise",
        options=["MAE", "RMSE", "R²"],
    )

    metric_figure = create_model_metrics_chart(
        selected_metric
    )

    st.plotly_chart(
        metric_figure,
        width="stretch",
        key=f"selected_metric_chart_{selected_metric}",
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        mae_figure = create_model_metrics_chart(
            "MAE"
        )

        st.plotly_chart(
            mae_figure,
            width="stretch",
            key="model_comparison_mae_chart",
        )

    with chart_col2:

        r2_figure = create_model_metrics_chart(
            "R²"
        )

        st.plotly_chart(
            r2_figure,
            width="stretch",
            key="model_comparison_r2_chart",
        )

    st.subheader("Model Selection")

    st.success(
        """
        **Linear Regression was selected as the final model.**

        It achieved the lowest MAE and RMSE and the highest R² score
        among the evaluated models.
        """
    )

    st.markdown(
        """
        ### Why did the more complex models underperform?

        Random Forest and XGBoost can learn nonlinear relationships,
        but additional complexity does not automatically guarantee
        better forecasting performance.

        Possible reasons include:

        - The relatively small training dataset.
        - High variability and extreme sales values.
        - Limited predictable structure in the available features.
        - Model sensitivity to unusual observations.
        - The current hyperparameter configurations.
        - A single train-test split rather than extensive backtesting.

        Selecting Linear Regression is therefore an evidence-based
        decision rather than an assumption that the most complex
        algorithm must perform best.
        """
    )


# =========================================================
# MODEL EXPLAINABILITY PAGE
# =========================================================

elif page == "🧠 Model Explainability":

    st.header("🧠 Model Explainability")

    st.markdown(
        """
        Analyse which engineered features have the greatest influence
        on each forecasting model.

        Linear Regression uses coefficient magnitude, while Random
        Forest and XGBoost provide tree-based feature importance.
        """
    )

    available_explainability_models = [
        model_name
        for model_name in MODEL_PATHS
        if model_name in forecasting_models
    ]

    if not available_explainability_models:
        st.error(
            "No loaded model is available for explainability analysis."
        )
        st.stop()

    selected_explanation_model = st.selectbox(
        "Select a model to explain",
        options=available_explainability_models,
    )

    selected_model = forecasting_models[
        selected_explanation_model
    ]

    explanation_df = get_model_feature_importance(
        selected_explanation_model,
        selected_model,
    )

    if explanation_df is None:
        st.warning(
            "Feature importance could not be extracted from this model."
        )

    else:

        most_important_feature = (
            explanation_df.iloc[0]["Feature"]
        )

        most_important_value = float(
            explanation_df.iloc[0]["Importance"]
        )

        explanation_col1, explanation_col2, explanation_col3 = (
            st.columns(3)
        )

        explanation_col1.metric(
            "Explained Model",
            selected_explanation_model,
        )

        explanation_col2.metric(
            "Most Influential Feature",
            most_important_feature,
        )

        explanation_col3.metric(
            "Importance Magnitude",
            f"{most_important_value:,.4f}",
        )

        st.subheader(
            f"{selected_explanation_model} Feature Importance"
        )

        plot_df = explanation_df.sort_values(
            "Importance",
            ascending=True,
        )

        feature_figure = px.bar(
            plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                f"{selected_explanation_model} "
                "Feature Importance"
            ),
        )

        feature_figure.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Importance",
            yaxis_title="Feature",
        )

        st.plotly_chart(
            feature_figure,
            width="stretch",
            key=(
                "individual_feature_importance_"
                + selected_explanation_model
            ),
        )

        st.subheader("Feature Importance Table")

        st.dataframe(
            explanation_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Raw Value": (
                    st.column_config.NumberColumn(
                        "Raw Value",
                        format="%.6f",
                    )
                ),
                "Importance": (
                    st.column_config.NumberColumn(
                        "Importance",
                        format="%.6f",
                    )
                ),
            },
        )

        if selected_explanation_model == (
            "Linear Regression"
        ):
            st.info(
                """
                Positive coefficients increase the predicted sales
                value when the corresponding feature increases.
                Negative coefficients reduce the predicted value.

                Coefficient sizes must be interpreted carefully because
                the model features use different numerical scales.
                """
            )

        else:
            st.info(
                """
                Tree-based importance measures how strongly each
                feature contributed to decisions across the fitted
                trees. It does not indicate whether the feature had
                a positive or negative effect.
                """
            )

    st.divider()
    st.subheader("Cross-Model Importance Comparison")

    normalized_comparison = (
        build_normalized_explainability_table(
            forecasting_models
        )
    )

    model_columns = [
        model_name
        for model_name in MODEL_PATHS
        if (
            model_name in normalized_comparison.columns
            and normalized_comparison[
                model_name
            ].notna().any()
        )
    ]

    if not model_columns:
        st.warning(
            "Cross-model feature comparison is not available."
        )

    else:

        st.markdown(
            """
            Importance values are normalised separately for each model.
            A value of **1.0** represents the most influential feature
            within that model.

            Normalisation allows the three models to be compared even
            though their raw importance scales are different.
            """
        )

        st.dataframe(
            normalized_comparison[
                [
                    "Feature",
                    *model_columns,
                    "Average Importance",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                model_name: (
                    st.column_config.NumberColumn(
                        model_name,
                        format="%.4f",
                    )
                )
                for model_name in [
                    *model_columns,
                    "Average Importance",
                ]
            },
        )

        comparison_long = (
            normalized_comparison[
                ["Feature", *model_columns]
            ]
            .melt(
                id_vars="Feature",
                var_name="Model",
                value_name="Normalized Importance",
            )
        )

        comparison_figure = px.bar(
            comparison_long,
            x="Feature",
            y="Normalized Importance",
            color="Model",
            barmode="group",
            title=(
                "Normalised Feature Importance "
                "Across Models"
            ),
        )

        comparison_figure.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Feature",
            yaxis_title="Normalised Importance",
            xaxis_tickangle=-45,
        )

        st.plotly_chart(
            comparison_figure,
            width="stretch",
            key="normalized_model_explainability_chart",
        )

        consensus_features = (
            normalized_comparison.sort_values(
                "Average Importance",
                ascending=False,
            )
            .head(5)["Feature"]
            .tolist()
        )

        st.subheader("Business Interpretation")

        st.success(
            """
            The explainability analysis shows that the forecasting
            models do not rely on the same variables in exactly the
            same way. This is expected because each algorithm learns
            relationships differently.
            """
        )

        st.markdown(
            f"""
            **Most consistently influential features**

            {", ".join(consensus_features)}

            Key findings:

            - Historical lag and rolling-average features represent
              recent customer-demand behaviour.
            - Calendar features capture weekday, monthly, quarterly,
              and seasonal sales patterns.
            - Linear Regression gives direct signed coefficients.
            - Random Forest captures nonlinear interactions through
              multiple decision trees.
            - XGBoost distributes importance across temporal and
              historical features through sequential boosting.
            - Features that appear important across several models
              are strong candidates for future monitoring and model
              improvement.
            """
        )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About the Project")

    st.markdown(
        """
        ## Sales Forecasting and Demand Intelligence

        This project develops an end-to-end machine learning system
        for analysing historical retail sales and generating future
        sales estimates.

        The application combines data preparation, exploratory
        analysis, time-series feature engineering, model training,
        evaluation, explainability, interactive forecasting, and
        dashboard development.
        """
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Analyse historical sales patterns and seasonal behaviour.
        - Create calendar, lag, and rolling-average features.
        - Train and compare multiple regression algorithms.
        - Select the best-performing model using objective metrics.
        - Explain how each model uses the engineered features.
        - Build an interactive dashboard for business users.
        - Provide downloadable datasets and prediction results.
        """
    )

    st.subheader("Machine Learning Models")

    model_col1, model_col2, model_col3 = (
        st.columns(3)
    )

    with model_col1:
        st.markdown(
            """
            ### Linear Regression

            An interpretable baseline model that estimates sales
            using a weighted linear combination of input features.

            **Final status:** Selected best-performing model.
            """
        )

    with model_col2:
        st.markdown(
            """
            ### Random Forest

            An ensemble model combining multiple decision trees to
            capture nonlinear relationships and feature interactions.

            **Final status:** Evaluated as an alternative model.
            """
        )

    with model_col3:
        st.markdown(
            """
            ### XGBoost

            A gradient-boosting algorithm that sequentially improves
            decision trees to reduce prediction error.

            **Final status:** Evaluated as an alternative model.
            """
        )

    st.subheader("Engineered Features")

    feature_description = pd.DataFrame(
        {
            "Feature Group": [
                "Calendar features",
                "Lag features",
                "Rolling features",
                "Target",
            ],
            "Features": [
                (
                    "Year, Month, Quarter, Week, Day, "
                    "DayOfWeek, DayOfYear, IsWeekend"
                ),
                "Lag_1, Lag_7, Lag_30",
                "Rolling_Mean_7, Rolling_Mean_30",
                "Sales",
            ],
            "Purpose": [
                (
                    "Represent calendar patterns "
                    "and seasonality."
                ),
                (
                    "Represent previous sales "
                    "behaviour."
                ),
                (
                    "Represent recent average "
                    "demand."
                ),
                (
                    "Value predicted by the machine "
                    "learning models."
                ),
            ],
        }
    )

    st.dataframe(
        feature_description,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Technology Stack")

    technology_data = pd.DataFrame(
        {
            "Category": [
                "Programming",
                "Data processing",
                "Machine learning",
                "Visualisation",
                "Application",
                "Version control",
            ],
            "Technology": [
                "Python",
                "Pandas and NumPy",
                "Scikit-learn and XGBoost",
                "Plotly and Matplotlib",
                "Streamlit",
                "Git and GitHub",
            ],
        }
    )

    st.dataframe(
        technology_data,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Project Workflow")

    st.markdown(
        """
        1. Raw retail sales data was loaded and cleaned.
        2. Daily, weekly, monthly, and yearly datasets were created.
        3. Exploratory analysis identified important sales patterns.
        4. Time, lag, and rolling-average features were engineered.
        5. Linear Regression, Random Forest, and XGBoost were trained.
        6. Models were evaluated using MAE, RMSE, and R².
        7. Linear Regression was selected as the final model.
        8. Feature importance was analysed across all three models.
        9. An interactive Streamlit application was developed.
        10. Forecast results and prediction history were made downloadable.
        """
    )

    st.subheader("Current Limitations")

    st.markdown(
        """
        - The dataset contains only 1,200 forecasting observations.
        - Large sales spikes remain difficult to predict.
        - Promotions, holidays, pricing, inventory, and economic
          variables are not currently included.
        - The dashboard generates point estimates rather than
          calibrated confidence intervals.
        - Forecast quality depends strongly on the historical values
          provided by the user.
        - Model metrics are based on the existing evaluation split
          rather than repeated chronological backtesting.
        """
    )

    st.subheader("Future Improvements")

    st.markdown(
        """
        - Add holiday, promotion, price, and inventory features.
        - Use rolling chronological backtesting.
        - Perform systematic hyperparameter tuning.
        - Add SHAP-based local prediction explanations.
        - Create multi-day recursive forecasts.
        - Add uncertainty intervals.
        - Introduce automated model retraining.
        - Connect the dashboard to a live sales database.
        - Add model and data-drift monitoring.
        - Deploy the application using Streamlit Community Cloud.
        """
    )

    st.divider()

    st.caption(
        "Developed as an end-to-end machine learning portfolio project."
    )