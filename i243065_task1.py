import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Exploratory Data Analysis Interface")
st.sidebar.header("Dataset Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"The uploaded file could not be read as a valid CSV: {e}")
        st.stop()
    st.subheader("Dataset Preview & Metadata")
    st.write("--First 5 Rows:")
    st.dataframe(df.head())
    st.write("--Shape:", df.shape)
    st.write("--Column Data Types:")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))
    st.write("--Missing Values Per Column: ")

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_percent
    })
    st.dataframe(missing_df)

    st.write("--Basic Numerical Statistics:")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        stats = numeric_df.describe().loc[["mean", "50%", "min", "max"]]
        stats = stats.rename(index={"50%": "median"})
        st.dataframe(stats)
    else:
        st.info("No numerical columns found in this dataset.")

    st.sidebar.header("Attribute Selection")
    selected_column = st.sidebar.selectbox("Select Attribute for Visualization", df.columns)

    if pd.api.types.is_numeric_dtype(df[selected_column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    st.subheader("Visualization")

    if column_type == "Numerical":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        value_counts = df[selected_column].value_counts()
        n_unique = value_counts.shape[0]
        MAX_CATEGORIES = 15

        if n_unique > MAX_CATEGORIES:
            st.warning(
                f"'{selected_column}' has {n_unique} unique values, which is too many "
                f"to plot individually (likely an identifier-like column, e.g. Name, "
                f"Ticket, or Cabin). Showing the top {MAX_CATEGORIES} most frequent "
                f"values grouped with an 'Other' category."
            )
            top_counts = value_counts.head(MAX_CATEGORIES)
            other_count = value_counts.iloc[MAX_CATEGORIES:].sum()
            if other_count > 0:
                top_counts = pd.concat([top_counts, pd.Series({"Other": other_count})])
            plot_counts = top_counts
        else:
            plot_counts = value_counts

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Bar Chart of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")

        total = plot_counts.sum()
        for i, v in enumerate(plot_counts):
            pct = (v / total) * 100
            ax.text(i, v, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")