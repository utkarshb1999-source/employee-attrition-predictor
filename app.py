import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("attrition_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
drivers_df = pd.read_csv("attrition_drivers.csv")

st.title("📊 Employee Attrition Risk Predictor")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    distance = st.number_input("Distance From Home", 1, 50, 10)
    monthly_income = st.number_input("Monthly Income", 1000, 50000, 10000)
num_companies_worked = st.number_input("Num Companies Worked", 0, 20, 2)
percent_salary_hike = st.number_input("Percent Salary Hike", 0, 100, 15)
training_times_last_year = st.number_input("Training Times Last Year", 0, 20, 2)
years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
overtime_days_pct = st.slider(
    "Overtime Days %",
    0.0,
    1.0,
    0.10
)
business_travel = st.selectbox(
    "Business Travel",
    [
        "Non-Travel",
        "Travel_Rarely",
        "Travel_Frequently"
    ]
)
department = st.selectbox(
    "Department",
    [
        "Human Resources",
        "Research & Development",
        "Sales"
    ]
)

education_field = st.selectbox(
    "Education Field",
    [
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "Female",
        "Male"
    ]
)

with col2:
    total_working_years = st.number_input("Total Working Years", 0, 50, 10)
    years_at_company = st.number_input("Years At Company", 0, 40, 5)
    years_with_curr_manager = st.number_input("Years With Current Manager", 0, 30, 3)
st.write("Model expects", len(feature_columns), "features")
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
job_involvement = st.slider("Job Involvement", 1, 4, 3)
performance_rating = st.slider("Performance Rating", 1, 4, 3)
avg_hours_per_day = st.slider(
    "Average Hours Per Day",
    4.0,
    12.0,
    8.0
)

attendance_rate = st.slider(
    "Attendance Rate",
    0.50,
    1.00,
    0.95
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Divorced",
        "Married",
        "Single"
    ]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative"
    ]
)

if st.button("Predict Attrition Risk"):

    input_df = pd.DataFrame([{
        "Age": age,
        "DistanceFromHome": distance,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "YearsAtCompany": years_at_company,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "JobInvolvement": job_involvement,
        "PerformanceRating": performance_rating,
        "avg_hours_per_day": avg_hours_per_day,
        "attendance_rate": attendance_rate,
        "overtime_days_pct": overtime_days_pct,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status
    }])

    st.write(input_df)

    encoded_df = pd.get_dummies(input_df)

    encoded_df = encoded_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    st.write("Encoded shape:", encoded_df.shape)

    scaled_data = scaler.transform(encoded_df)

    probability = model.predict_proba(scaled_data)[0][1]

    st.write("Attrition Probability:", round(probability * 100, 2), "%")

    risk_pct = probability * 100

    if risk_pct >= 60:
        st.error(f"🔴 High Risk: {risk_pct:.2f}%")

    elif risk_pct >= 30:
        st.warning(f"🟠 Medium Risk: {risk_pct:.2f}%")

    else:
        st.success(f"🟢 Low Risk: {risk_pct:.2f}%")

    st.progress(min(int(risk_pct), 100))

    st.metric(
        label="Attrition Risk Score",
        value=f"{risk_pct:.2f}%"
    )
    st.subheader("Top Attrition Drivers")

    top_drivers = drivers_df.sort_values(
        by="Impact",
        ascending=False
    ).head(3)

    st.dataframe(top_drivers)
st.markdown("---")

st.caption(
    "Model: Logistic Regression | ROC-AUC: 0.8083 | Dataset: 4,410 Employees"
)
