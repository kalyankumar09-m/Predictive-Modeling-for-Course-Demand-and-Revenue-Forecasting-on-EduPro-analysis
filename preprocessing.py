import pandas as pd

FILE_PATH = "EduPro Online Platform.xlsx"


def load_sheets():
    """
    Load all Excel sheets safely.
    """

    excel_file = pd.ExcelFile(FILE_PATH)

    print("Available Sheets:", excel_file.sheet_names)

    courses = pd.read_excel(FILE_PATH, sheet_name="Courses")
    teachers = pd.read_excel(FILE_PATH, sheet_name="Teachers")
    transactions = pd.read_excel(FILE_PATH, sheet_name="Transactions")

    return courses, teachers, transactions



def preprocess_data():
    """
    Clean and merge all datasets.
    """

    courses, teachers, transactions = load_sheets()

    # -----------------------------
    # CLEAN COLUMN NAMES
    # -----------------------------
    courses.columns = courses.columns.str.strip()
    teachers.columns = teachers.columns.str.strip()
    transactions.columns = transactions.columns.str.strip()

    # -----------------------------
    # REQUIRED COLUMNS CHECK
    # -----------------------------
    required_course_cols = [
        "CourseID",
        "CourseCategory",
        "CourseType",
        "CourseLevel",
        "CoursePrice",
        "CourseDuration",
        "CourseRating",
    ]

    required_teacher_cols = [
        "TeacherID",
        "Expertise",
        "YearsOfExperience",
        "TeacherRating",
    ]

    required_transaction_cols = [
        "TransactionID",
        "CourseID",
        "Amount",
    ]

    for col in required_course_cols:
        if col not in courses.columns:
            raise Exception(f"Missing column in Courses sheet: {col}")

    for col in required_teacher_cols:
        if col not in teachers.columns:
            raise Exception(f"Missing column in Teachers sheet: {col}")

    for col in required_transaction_cols:
        if col not in transactions.columns:
            raise Exception(f"Missing column in Transactions sheet: {col}")

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    courses["CourseRating"] = courses["CourseRating"].fillna(
        courses["CourseRating"].mean()
    )

    teachers["TeacherRating"] = teachers["TeacherRating"].fillna(
        teachers["TeacherRating"].mean()
    )

    teachers["YearsOfExperience"] = teachers["YearsOfExperience"].fillna(
        teachers["YearsOfExperience"].median()
    )

    # -----------------------------
    # AGGREGATE TRANSACTION DATA
    # -----------------------------
    revenue_data = transactions.groupby("CourseID").agg(
        EnrollmentCount=("TransactionID", "count"),
        Revenue=("Amount", "sum"),
        AverageTransaction=("Amount", "mean"),
    ).reset_index()

    # -----------------------------
    # MERGE COURSES + REVENUE
    # -----------------------------
    df = pd.merge(
        courses,
        revenue_data,
        on="CourseID",
        how="left"
    )

    # -----------------------------
    # MERGE TEACHERS
    # -----------------------------
    if "TeacherID" in courses.columns:
        df = pd.merge(
            df,
            teachers,
            on="TeacherID",
            how="left"
        )

    # -----------------------------
    # FILL NULLS AFTER MERGE
    # -----------------------------
    numeric_cols = [
        "EnrollmentCount",
        "Revenue",
        "AverageTransaction",
        "TeacherRating",
        "YearsOfExperience",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------

    # Price bands
    df["PriceBand"] = pd.cut(
        df["CoursePrice"],
        bins=[0, 50, 150, 1000],
        labels=["Low", "Medium", "High"],
    )

    # Duration buckets
    df["DurationBucket"] = pd.cut(
        df["CourseDuration"],
        bins=[0, 5, 20, 200],
        labels=["Short", "Medium", "Long"],
    )

    # Rating tier
    df["RatingTier"] = pd.cut(
        df["CourseRating"],
        bins=[0, 2, 4, 5],
        labels=["Poor", "Good", "Excellent"],
    )

    # Experience bucket
    if "YearsOfExperience" in df.columns:
        df["ExperienceBucket"] = pd.cut(
            df["YearsOfExperience"],
            bins=[0, 2, 5, 50],
            labels=["Beginner", "Intermediate", "Expert"],
        )

    return df
