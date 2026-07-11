
import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# -------------------------------------------------
# Load Trained Model from Hugging Face
# -------------------------------------------------
model_path = hf_hub_download(
    repo_id="<your_username>/visit-with-us-model",
    filename="visit_with_us_model.joblib"
)

model = joblib.load(model_path)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Visit with Us - Wellness Tourism Package Prediction")

st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package** based on customer demographics, travel history, and sales interaction details.
""")

# -------------------------------------------------
# Customer Information
# -------------------------------------------------

Age = st.number_input("Age", min_value=18, max_value=100, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1,2,3]
)

Occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

Gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star",
    [3,4,5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

NumberOfTrips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    max_value=20,
    value=2
)

Passport = st.selectbox(
    "Passport Available",
    [0,1]
)

OwnCar = st.selectbox(
    "Own Car",
    [0,1]
)

NumberOfChildrenVisiting = st.number_input(
    "Children Visiting",
    min_value=0,
    max_value=5,
    value=0
)

Designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=30000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    1,
    5,
    3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

NumberOfFollowups = st.number_input(
    "Number of Follow Ups",
    min_value=0,
    max_value=10,
    value=2
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (Minutes)",
    min_value=5,
    max_value=120,
    value=20
)

# -------------------------------------------------
# Create DataFrame
# -------------------------------------------------

input_data = pd.DataFrame({

    "Age":[Age],
    "TypeofContact":[TypeofContact],
    "CityTier":[CityTier],
    "Occupation":[Occupation],
    "Gender":[Gender],
    "NumberOfPersonVisiting":[NumberOfPersonVisiting],
    "PreferredPropertyStar":[PreferredPropertyStar],
    "MaritalStatus":[MaritalStatus],
    "NumberOfTrips":[NumberOfTrips],
    "Passport":[Passport],
    "OwnCar":[OwnCar],
    "NumberOfChildrenVisiting":[NumberOfChildrenVisiting],
    "Designation":[Designation],
    "MonthlyIncome":[MonthlyIncome],
    "PitchSatisfactionScore":[PitchSatisfactionScore],
    "ProductPitched":[ProductPitched],
    "NumberOfFollowups":[NumberOfFollowups],
    "DurationOfPitch":[DurationOfPitch]

})

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"Customer is likely to purchase the Wellness Tourism Package.\n\n"
            f"Probability: **{probability:.2%}**"
        )

    else:

        st.error(
            f"Customer is unlikely to purchase the Wellness Tourism Package.\n\n"
            f"Probability: **{probability:.2%}**"
        )
