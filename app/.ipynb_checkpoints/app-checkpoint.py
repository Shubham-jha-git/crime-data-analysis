import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

st.title("Crime Data Analysis & Prediction App")

st.markdown(
    """
    ### 🧠 Crime Prediction Dashboard
    This application predicts crime counts based on **State**, **Crime Type**, and **Year**  
    using a trained Machine Learning model.
    """
)


# Debug: show working directory
st.write("Working directory:", os.getcwd())

# Load cleaned data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/crime_data_cleaned.csv")

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/crime_prediction_model.pkl")

model = load_model()

st.success("Data and ML model loaded successfully!")

st.sidebar.header("🔎 Select Parameters")

state_list = sorted(df["State"].unique())
selected_state = st.sidebar.selectbox("Select State", state_list)

crime_list = sorted(df["Crime_Type"].unique())
crime_list.insert(0, "All Crimes (Combined)")
selected_crime = st.sidebar.selectbox("Select Crime Type", crime_list)

selected_year = st.sidebar.number_input(
    "Select Year",
    min_value=int(df["Year"].min()),
    max_value=2050,
    value=2030,
    step=1
)



st.write("### Selected Inputs")
st.write("State:", selected_state)
st.write("Crime Type:", selected_crime)
st.write("Year:", selected_year)

st.divider()

# Prediction button
st.divider()

if st.button("Predict Crime Count"):

    # 1️⃣ Create input dataframe with all zeros
    input_df = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)

    # 2️⃣ Set Year
    input_df["Year"] = selected_year

    # 3️⃣ Set selected State
    state_col = f"State_{selected_state}"
    if state_col in input_df.columns:
        input_df[state_col] = 1

    # 4️⃣ Set Crime Type(s)
    if selected_crime == "All Crimes (Combined)":
        # Turn ON all crime type columns
        for col in input_df.columns:
            if col.startswith("Crime_Type_"):
                input_df[col] = 1
    else:
        crime_col = f"Crime_Type_{selected_crime}"
        if crime_col in input_df.columns:
            input_df[crime_col] = 1

    # 5️⃣ Predict
    prediction = model.predict(input_df)[0]

    st.success(f"🔮 Predicted Crime Count: {int(prediction)}")

#Graphs
st.subheader("📊 Crime Trend Over the Years")

state_df = df[df["State"] == selected_state]

if selected_crime == "All Crimes (Combined)":
    trend_df = (
        state_df
        .groupby("Year")["Crime_Count"]
        .sum()
        .reset_index()
    )
else:
    trend_df = (
        state_df[state_df["Crime_Type"] == selected_crime]
        .groupby("Year")["Crime_Count"]
        .sum()
        .reset_index()
    )

# Smoothing using rolling average
trend_df["Smoothed_Crime"] = trend_df["Crime_Count"].rolling(window=3, min_periods=1).mean()

fig, ax = plt.subplots()
ax.plot(trend_df["Year"], trend_df["Crime_Count"], label="Actual", marker="o")
ax.plot(trend_df["Year"], trend_df["Smoothed_Crime"], label="Trend (Smoothed)", linewidth=2)

ax.set_xlabel("Year")
ax.set_ylabel("Crime Count")
ax.set_title(f"Crime Trend in {selected_state}")
ax.grid(True)
ax.legend()

st.pyplot(fig)
st.caption(
    "This graph shows how crime levels have changed year-wise in the selected state. "
    "The smoothed line highlights the overall trend by reducing short-term fluctuations."
)



#Comparison Graph
st.subheader("📊 Crime Type Comparison")

# Aggregate crime count by crime type
comparison_df = (
    df[df["State"] == selected_state]
    .groupby("Crime_Type")["Crime_Count"]
    .sum()
    .reset_index()
)

fig2, ax2 = plt.subplots()
ax2.bar(comparison_df["Crime_Type"], comparison_df["Crime_Count"])
ax2.set_xlabel("Crime Type")
ax2.set_ylabel("Total Crime Count")
ax2.set_title(f"Crime Distribution by Type in {selected_state}")
ax2.tick_params(axis='x', rotation=45)

st.pyplot(fig2)
st.caption(
    "This graph compares different crime categories based on their total occurrence. "
    "It helps identify which types of crimes are more prevalent in the selected state."
)
