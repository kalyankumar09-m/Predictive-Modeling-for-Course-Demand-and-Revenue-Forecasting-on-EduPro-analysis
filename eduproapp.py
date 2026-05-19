import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import train_model


# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="EduPro Dashboard",
    layout="wide",
)

st.title("EduPro Predictive Analytics Dashboard")

# ---------------------------------
# LOAD MODEL
# ---------------------------------
model, features, df = train_model()

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.header("Course Input")

course_price = st.sidebar.slider(
    "Course Price",
    10,
    1000,
    100
)

course_duration = st.sidebar.slider(
    "Course Duration",
    1,
    100,
    20
)

course_rating = st.sidebar.slider(
    "Course Rating",
    1.0,
    5.0,
    4.0
)

teacher_rating = st.sidebar.slider(
    "Teacher Rating",
    1.0,
    5.0,
    4.0
)

experience = st.sidebar.slider(
    "Years Of Experience",
    0,
    30,
    5
)

# ---------------------------------
# INPUT DATA
# ---------------------------------
input_data = {
    "CoursePrice": [course_price],
    "CourseDuration": [course_duration],
    "CourseRating": [course_rating],
    "TeacherRating": [teacher_rating],
    "YearsOfExperience": [experience],
}

# Add missing feature columns
for feature in features:

    if feature not in input_data:
        input_data[feature] = [0]

sample_df = pd.DataFrame(input_data)

sample_df = sample_df[features]

# ---------------------------------
# PREDICTION
# ---------------------------------
prediction = model.predict(sample_df)[0]

# ---------------------------------
# SHOW PREDICTION
# ---------------------------------
st.metric(
    label="Predicted Revenue",
    value=f"${prediction:,.2f}"
)

# ---------------------------------
# DATA PREVIEW
# ---------------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

# ---------------------------------
# CATEGORY REVENUE
# ---------------------------------
st.subheader("Revenue By Category")

if "CourseCategory" in df.columns:

    category_revenue = (
        df.groupby("CourseCategory")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    category_revenue.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Revenue")

    st.pyplot(fig)

# ---------------------------------
# FEATURE IMPORTANCE
# ---------------------------------
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance_df)

# ---------------------------------
# TOP COURSES
# ---------------------------------
st.subheader("Top Revenue Courses")

if "Revenue" in df.columns:

    top_courses = df.sort_values(
        by="Revenue",
        ascending=False
    )

    cols_to_show = [
        col for col in [
            "CourseID",
            "CourseCategory",
            "Revenue",
            "EnrollmentCount",
        ]
        if col in top_courses.columns
    ]

    st.dataframe(
        top_courses[cols_to_show].head(10)
    )

# ---------------------------------
# DOWNLOAD DATA
# ---------------------------------
csv = df.to_csv(index=False)

st.download_button(
    label="Download Processed Data",
    data=csv,
    file_name="processed_edupro_data.csv",
    mime="text/csv",
)