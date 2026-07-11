from datetime import date
from pathlib import Path
import pickle

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

# Metrics obtained from the model-training notebooks.
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

    required_columns = ["Order Date", "Sales", *FEATURE_COLUMNS]

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
    """
    Load available forecasting models.

    Returns:
        loaded_models: Models successfully loaded.
        model_errors: Errors for models that could not be loaded.
    """

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
    st.error("The dashboard could not load the forecasting dataset.")
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
    """Format a value as currency."""

    return f"${value:,.2f}"


def build_feature_row(
    forecast_date: date,
    lag_1: float,
    lag_7: float,
    lag_30: float,
    rolling_mean_7: float,
    rolling_mean_30: float,
) -> pd.DataFrame:
    """Create the exact feature structure used by the models."""

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
    """
    Align feature order with the saved model when feature names
    are available.
    """

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_data.columns
        ]

        if missing_features:
            raise ValueError(
                "The model expects features that are not available: "
                + ", ".join(missing_features)
            )

        return input_data[expected_features]

    return input_data[FEATURE_COLUMNS]


def calculate_input_reliability(
    input_data: pd.DataFrame,
    training_data: pd.DataFrame,
) -> tuple[str, str]:
    """
    Provide an input-range reliability label.

    This is not a statistical confidence interval. It indicates
    whether the user inputs are within ranges seen during training.
    """

    historical_features = [
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30",
    ]

    outside_range_count = 0

    for feature in historical_features:
        value = float(input_data.iloc[0][feature])
        training_min = float(training_data[feature].min())
        training_max = float(training_data[feature].max())

        if value < training_min or value > training_max:
            outside_range_count += 1

    if outside_range_count == 0:
        return (
            "High input reliability",
            "All historical inputs are within ranges observed "
            "in the training dataset.",
        )

    if outside_range_count <= 2:
        return (
            "Moderate input reliability",
            f"{outside_range_count} historical input value(s) are "
            "outside the ranges observed during training.",
        )

    return (
        "Low input reliability",
        f"{outside_range_count} historical input values are outside "
        "the ranges observed during training. Treat the forecast "
        "with additional caution.",
    )


def create_forecast_comparison_chart(
    predicted_sales: float,
    historical_average: float,
    rolling_average: float,
) -> go.Figure:
    """Create a comparison chart for the generated prediction."""

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
        text_auto=".2s",
        title="Forecast Context Comparison",
    )

    figure.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="",
        yaxis_title="Sales",
    )

    return figure


def create_model_metrics_chart(
    metric: str,
) -> go.Figure:
    """Create a model-comparison chart."""

    ascending = metric != "R²"

    chart_data = MODEL_METRICS.sort_values(
        metric,
        ascending=ascending,
    )

    figure = px.bar(
        chart_data,
        x="Model",
        y=metric,
        text_auto=False,
        title=f"Model Comparison by {metric}",
    )
    
    figure.update_traces(
        texttemplate="%{y:,.2f}",
        textposition="outside",
    )

    figure.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="",
        yaxis_title=metric,
    )

    return figure


def get_best_model_name() -> str:
    """Return the best model using highest R²."""

    best_row = MODEL_METRICS.sort_values(
        "R²",
        ascending=False,
    ).iloc[0]

    return str(best_row["Model"])


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "Sales Trends",
        "Forecast",
        "Model Comparison",
        "About Project",
    ],
)

st.sidebar.divider()

available_model_count = len(forecasting_models)

st.sidebar.metric(
    "Available Models",
    f"{available_model_count}/{len(MODEL_PATHS)}",
)

if model_loading_errors:
    with st.sidebar.expander("Model loading information"):

        for model_name, error_message in (
            model_loading_errors.items()
        ):
            st.warning(
                f"**{model_name}:** {error_message}"
            )

st.sidebar.divider()

st.sidebar.caption(
    "Sales Forecasting and Demand Intelligence Project"
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
# DATASET OVERVIEW PAGE
# =========================================================

if page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    st.markdown(
        """
        This section provides an overview of the processed sales
        forecasting dataset used for model development and evaluation.
        """
    )

    total_records = len(forecast_df)
    total_features = forecast_df.shape[1]

    start_date = forecast_df["Order Date"].min().date()
    end_date = forecast_df["Order Date"].max().date()

    average_sales = forecast_df["Sales"].mean()
    median_sales = forecast_df["Sales"].median()
    maximum_sales = forecast_df["Sales"].max()
    minimum_sales = forecast_df["Sales"].min()

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    row1_col1.metric(
        "Total Records",
        f"{total_records:,}",
    )

    row1_col2.metric(
        "Total Features",
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

elif page == "Sales Trends":

    st.header("📈 Sales Trend Analysis")

    st.markdown(
        """
        Explore daily, monthly, yearly, weekday, and seasonal sales
        patterns using interactive visualizations.
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

    if isinstance(selected_date_range, tuple) and (
        len(selected_date_range) == 2
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

    # Daily sales
    st.subheader("Daily Sales")

    daily_sales = (
        filtered_df.groupby("Order Date", as_index=False)["Sales"]
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
    )

    st.plotly_chart(
        fig_daily,
        width="stretch",
    )

    # Monthly sales
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
    )

    st.plotly_chart(
        fig_monthly,
        width="stretch",
    )

    # Yearly sales
    st.subheader("Yearly Sales")

    yearly_sales = (
        filtered_df.groupby("Year", as_index=False)["Sales"]
        .sum()
    )

    yearly_sales["Year"] = yearly_sales["Year"].astype(str)

    fig_yearly = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        text_auto=".2s",
        title="Yearly Sales",
    )

    fig_yearly.update_layout(
        template="plotly_dark",
        height=450,
    )

    st.plotly_chart(
        fig_yearly,
        width="stretch",
    )

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:

        weekday_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }

        weekday_sales = (
            filtered_df.groupby(
                "DayOfWeek",
                as_index=False,
            )["Sales"]
            .mean()
        )

        weekday_sales["Weekday"] = (
            weekday_sales["DayOfWeek"].map(weekday_names)
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
        )

        st.plotly_chart(
            fig_weekday,
            width="stretch",
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
            "Q" + quarter_sales["Quarter"].astype(str)
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
        )

        st.plotly_chart(
            fig_quarter,
            width="stretch",
        )


# =========================================================
# FORECAST PAGE
# =========================================================

elif page == "Forecast":

    st.header("🔮 Interactive Sales Forecast")

    st.markdown(
        """
        Select a forecasting model, choose a date, and provide recent
        historical sales information to generate an estimated sales value.
        """
    )

    if not forecasting_models:
        st.error(
            "No trained forecasting models are currently available. "
            "Make sure the `.pkl` model files exist inside the models folder."
        )
        st.stop()

    st.info(
        "Linear Regression currently produced the best evaluation results "
        "among the three trained models."
    )

    available_model_names = list(forecasting_models.keys())

    default_model_index = (
        available_model_names.index("Linear Regression")
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

        selected_timestamp = pd.Timestamp(forecast_date)

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
                    str(int(selected_timestamp.isocalendar().week)),
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

        with st.expander("View automatically generated calendar features"):
            st.dataframe(
                derived_calendar_data,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Historical Sales Features")

        st.caption(
            "Enter recent sales values. Default values are based on "
            "the median values in the training dataset."
        )

        input_col1, input_col2 = st.columns(2)

        with input_col1:

            lag_1 = st.number_input(
                "Previous Sales Value — Lag 1",
                min_value=0.0,
                value=float(forecast_df["Lag_1"].median()),
                step=100.0,
                format="%.2f",
                help="Sales recorded one observation before the forecast.",
            )

            lag_7 = st.number_input(
                "Sales 7 Observations Ago — Lag 7",
                min_value=0.0,
                value=float(forecast_df["Lag_7"].median()),
                step=100.0,
                format="%.2f",
                help="Sales recorded seven observations before the forecast.",
            )

            lag_30 = st.number_input(
                "Sales 30 Observations Ago — Lag 30",
                min_value=0.0,
                value=float(forecast_df["Lag_30"].median()),
                step=100.0,
                format="%.2f",
                help="Sales recorded thirty observations before the forecast.",
            )

        with input_col2:

            rolling_mean_7 = st.number_input(
                "7-Observation Rolling Mean",
                min_value=0.0,
                value=float(
                    forecast_df["Rolling_Mean_7"].median()
                ),
                step=100.0,
                format="%.2f",
                help="Average sales across the previous seven observations.",
            )

            rolling_mean_30 = st.number_input(
                "30-Observation Rolling Mean",
                min_value=0.0,
                value=float(
                    forecast_df["Rolling_Mean_30"].median()
                ),
                step=100.0,
                format="%.2f",
                help="Average sales across the previous thirty observations.",
            )

        submitted = st.form_submit_button(
            "Generate Forecast",
            type="primary",
            width="stretch",
        )

    if submitted:

        selected_model = forecasting_models.get(model_name)

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

                raw_prediction = selected_model.predict(
                    model_input
                )[0]

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

                st.session_state.latest_prediction = {
                    "Forecast Date": str(forecast_date),
                    "Model": model_name,
                    "Predicted Sales": predicted_sales,
                    "Input Reliability": reliability_label,
                    **input_data.iloc[0].to_dict(),
                }

                st.session_state.prediction_history.append(
                    st.session_state.latest_prediction.copy()
                )

                st.success(
                    "Forecast generated successfully."
                )

            except Exception as error:
                st.error(
                    "The forecast could not be generated."
                )
                st.exception(error)


    # -----------------------------------------------------
    # Display latest forecast
    # -----------------------------------------------------

    if st.session_state.latest_prediction is not None:

        result = st.session_state.latest_prediction

        st.divider()

        st.subheader("Forecast Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Predicted Sales",
            format_currency(result["Predicted Sales"]),
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

        if reliability == "High input reliability":
            st.success(
                "High input reliability: all historical inputs are "
                "within ranges observed in the training dataset."
            )

        elif reliability == "Moderate input reliability":
            st.warning(
                "Moderate input reliability: some entered values are "
                "outside the ranges observed during training."
            )

        else:
            st.error(
                "Low input reliability: several entered values are "
                "outside the training-data ranges. Interpret this "
                "forecast cautiously."
            )

        comparison_figure = create_forecast_comparison_chart(
            predicted_sales=result["Predicted Sales"],
            historical_average=float(
                forecast_df["Sales"].mean()
            ),
            rolling_average=float(
                result["Rolling_Mean_30"]
            ),
        )

        st.plotly_chart(
            comparison_figure,
            width="stretch",
            key="forecast_comparison_chart"
        )

        with st.expander("View complete model input"):

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


    # -----------------------------------------------------
    # Prediction history
    # -----------------------------------------------------

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
        )

        download_col1, download_col2 = st.columns(2)

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

elif page == "Model Comparison":

    st.header("⚖️ Model Comparison")

    st.markdown(
        """
        Compare the forecasting models using the evaluation results
        obtained during model training.

        - **MAE:** Lower values indicate smaller average prediction errors.
        - **RMSE:** Lower values indicate better handling of large errors.
        - **R²:** Higher values indicate stronger explanatory performance.
        """
    )

    best_model_name = get_best_model_name()

    best_model_row = MODEL_METRICS.loc[
        MODEL_METRICS["Model"] == best_model_name
    ].iloc[0]

    best_col1, best_col2, best_col3, best_col4 = st.columns(4)

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

    comparison_table["Status"] = comparison_table["Model"].apply(
        lambda model: (
            "Selected best model"
            if model == best_model_name
            else "Alternative model"
        )
    )

    comparison_table["Model File"] = comparison_table["Model"].apply(
        lambda model: (
            "Available"
            if model in forecasting_models
            else "Unavailable"
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
        "Choose a metric to visualize",
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
        
        r2_figure.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
        )
        
        r2_figure.update_layout(
            yaxis_title="R²",
            yaxis_tickformat=".2f",
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

        Random Forest and XGBoost are capable of learning nonlinear
        relationships, but increased model complexity does not guarantee
        better performance. Their lower test results may be influenced by:

        - The relatively small training dataset.
        - High variability and extreme sales values.
        - Limited predictable structure in the available features.
        - Model sensitivity to outliers.
        - The current hyperparameter configuration.

        Selecting Linear Regression is therefore an evidence-based model
        selection decision rather than an assumption that the most complex
        algorithm must perform best.
        """
    )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.header("ℹ️ About the Project")

    st.markdown(
        """
        ## Sales Forecasting and Demand Intelligence

        This project develops an end-to-end machine learning system for
        analysing historical retail sales and generating future sales
        estimates.

        The application combines data preparation, exploratory analysis,
        time-series feature engineering, machine learning, model
        evaluation, interactive forecasting, and dashboard development.
        """
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Analyse historical sales patterns and seasonal behaviour.
        - Create calendar, lag, and rolling-average forecasting features.
        - Train and compare multiple regression algorithms.
        - Select the best-performing model using objective metrics.
        - Build an interactive dashboard for business users.
        - Provide downloadable data and forecast results.
        """
    )

    st.subheader("Machine Learning Models")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.markdown(
            """
            ### Linear Regression

            A simple and interpretable baseline model that estimates
            sales using a linear combination of the input features.

            **Final status:** Selected best model.
            """
        )

    with model_col2:
        st.markdown(
            """
            ### Random Forest

            An ensemble model that combines multiple decision trees
            to capture nonlinear relationships and feature interactions.

            **Final status:** Evaluated as an alternative model.
            """
        )

    with model_col3:
        st.markdown(
            """
            ### XGBoost

            A gradient-boosting model that sequentially improves weak
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
                "Represent calendar patterns and seasonality.",
                "Represent previous sales behaviour.",
                "Represent recent average demand.",
                "Value predicted by the machine learning models.",
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
        2. Daily, weekly, monthly, and yearly sales datasets were created.
        3. Exploratory analysis was conducted to identify sales patterns.
        4. Time-based, lag, and rolling-average features were engineered.
        5. Linear Regression, Random Forest, and XGBoost were trained.
        6. Models were evaluated using MAE, RMSE, and R².
        7. Linear Regression was selected as the final model.
        8. An interactive Streamlit dashboard was created.
        """
    )

    st.subheader("Current Limitations")

    st.markdown(
        """
        - The dataset contains only 1,200 forecasting observations.
        - Large sales spikes are difficult for the models to predict.
        - External factors such as promotions, holidays, inventory,
          pricing, and economic conditions are not included.
        - The dashboard generates single-point estimates rather than
          statistically calibrated confidence intervals.
        - Forecast quality depends heavily on the historical lag values
          entered by the user.
        """
    )

    st.subheader("Future Improvements")

    st.markdown(
        """
        - Add holiday, promotion, price, and inventory features.
        - Use chronological backtesting instead of one train-test split.
        - Perform systematic hyperparameter tuning.
        - Add SHAP-based model explainability.
        - Create multi-day recursive forecasts.
        - Add automated model retraining.
        - Connect the dashboard to a live sales database.
        - Deploy the application using Streamlit Community Cloud.
        """
    )

    st.divider()

    st.caption(
        "Developed as an end-to-end machine learning portfolio project."
    )
