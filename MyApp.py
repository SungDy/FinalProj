import pandas as pd
import streamlit as st
import altair as alt
import calendar

# Title of the web app
st.title('❤️‍🩹 Healthcare Stroke Dataset ')

# Adding some text to the app
st.write(
    'A Stroke is a medical condition that may cause sudden death. In these cases, the movement ability of the patient reduces or even stops.',
    'The patient may also experience problems in speaking and understanding languages. If not treated in time, it will cause death.',
    'In addition, stroke is correlated with high blood pressure, smoking, obesity, high blood cholesterol, and diabetes.',
    'In this work, we propose to use the database proposed in to identify people who are likely to be suffering a stroke.',
    'The final output is a classification/boolean problem.',
    'and also what specific crops suitable to be planted in an appropriate area'
)
st.write('Submitted by: Prince Naif Cambing BSIT-3A')

# Add a link to your GitHub account
st.markdown("[Visit my GitHub](https://github.com/SungDy)")

# Load dataset
df = pd.read_csv("New_healthcare-dataset-stroke-data.csv")
st.dataframe(df)

# Description
st.header('Primary Objective')
st.write('The primary objective of this notebook is to conduct a thorough evaluation of various machine learning models classifiers ',
         'to identify the most accurate algorithm for predicting the target variable. ',
         'The evaluation will culminate with the application of the best-performing model on a set of dummy data to demonstrate its predictive capabilities.')

st.header('Dataset Overview')

# Set up file upload
upload_file = st.sidebar.file_uploader(label="Upload your CSV or Excel file here", type=['csv', 'xlsx'])

def main():
    st.sidebar.title("Data Visualization")

    pages = {
        "Home": home,
        "Distribution of Gender Feature": distribution_of_gender_feature,
        "Stroke Distribution by Gender Group": stroke_distribution_by_gender_group,
        "Age Distribution": age_distribution,
        "Stroke Distribution by Age Group": stroke_distribution_by_age_group,
        "Age Distribution by Stroke Status": age_distribution_by_stroke_status,
        "Marriage Distribution by Stroke Status": marriage_distribution_by_stroke_status,
        "Work Type Distribution by Stroke Status": work_type_distribution_by_stroke_status,
        "Residence Type Distribution by Stroke Status": residence_type_distribution_by_stroke_status,
        "Residence Distribution by Stroke Status": residence_distribution_by_stroke_status,
        "Average Glucose Level by Age Group": average_glucose_level_by_age_group,
        "Smoking Status Distribution by Stroke": smoking_status_distribution_by_stroke,
        "Correlation Matrix Heatmap": correlation_matrix_heatmap,
        "Full Year Calendar": display_full_year_calendar  # Added the calendar function to pages
    }

    selection = st.sidebar.selectbox("Select Data", list(pages.keys()))

    # Display the selected page
    page = pages[selection]
    page()

    # Add a button to the sidebar to display the full year calendar
    if st.sidebar.button('Show Full Year Calendar'):
        display_full_year_calendar()

def home():
    st.header("Home")
    st.write("About Data")
    st.write("Dataset Overview")
    st.write(
        """
        According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally 
        responsible for approximately 11% of total deaths. This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender 
        age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.

        **Attributes Overview**:

        - id: unique identifier.
        - gender: "Male", "Female" or "Other".
        - age: age of the patient.
        - hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension.
        - heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease.
        - ever_married: "No" or "Yes".
        - work_type: "children", "Govt_job", "Never_worked", "Private" or "Self-employed".
        - residence_type: "Rural" or "Urban".
        - avg_glucose_level: average glucose level in blood.
        - bmi: body mass index.
        - smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown".
        - stroke: 1 if the patient had a stroke or 0 if not.
        """
    )

def distribution_of_gender_feature():
    st.header("Distribution of Gender Feature")
    
    # Check if the gender column exists
    if 'gender' not in df.columns:
        st.error('Gender column not found in the dataset')
        return
    
    # Create the Altair chart
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('gender', title='Gender'),
            y=alt.Y('count()', title='Count'),
            color='gender'
        ).properties(
            title='Distribution of Gender Feature'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def stroke_distribution_by_gender_group():
    st.header("Stroke Distribution by Gender Group")
    
    # Check if the necessary columns exist
    if 'gender' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    # Create the Altair chart
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('gender', title='Gender'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).transform_filter(
            alt.datum.stroke == 1
        ).properties(
            title='Stroke Distribution by Gender Group'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def age_distribution():
    st.header("Age Distribution")
    
    if 'age' not in df.columns:
        st.error('Age column not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('age', bin=True, title='Age'),
            y=alt.Y('count()', title='Count'),
            color=alt.condition(
                alt.datum.stroke == 1,
                alt.value('orange'),  # The color for stroke patients
                alt.value('steelblue')  # The color for non-stroke patients
            )
        ).properties(
            title='Age Distribution'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def stroke_distribution_by_age_group():
    st.header("Stroke Distribution by Age Group")
    
    if 'age' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('age', bin=True, title='Age'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).transform_filter(
            alt.datum.stroke == 1
        ).properties(
            title='Stroke Distribution by Age Group'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def age_distribution_by_stroke_status():
    st.header("Age Distribution by Stroke Status")
    
    if 'age' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('age', bin=True, title='Age'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).properties(
            title='Age Distribution by Stroke Status'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def marriage_distribution_by_stroke_status():
    st.header("Marriage Distribution by Stroke Status")
    
    if 'ever_married' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('ever_married', title='Marital Status'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).properties(
            title='Marriage Distribution by Stroke Status'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def work_type_distribution_by_stroke_status():
    st.header("Work Type Distribution by Stroke Status")
    
    if 'work_type' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('work_type', title='Work Type'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).properties(
            title='Work Type Distribution by Stroke'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def residence_type_distribution_by_stroke_status():
    st.header("Residence Type Distribution by Stroke Status")
    
    if 'residence_type' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('residence_type', title='Residence Type'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).properties(
            title='Residence Type Distribution by Stroke Status'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def residence_distribution_by_stroke_status():
    st.header("Residence Distribution by Stroke Status")
    
    if 'Residence' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Residence', title='Residence'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).properties(
            title='Residence Distribution by Stroke Status'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def average_glucose_level_by_age_group():
    st.header("Average Glucose Level by Age Group")
    
    if 'age' not in df.columns or 'avg_glucose_level' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('age', bin=True, title='Age'),
            y=alt.Y('mean(avg_glucose_level)', title='Average Glucose Level'),
            color='stroke:N'
        ).properties(
            title='Average Glucose Level by Age Group'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def smoking_status_distribution_by_stroke():
    st.header("Smoking Status Distribution by Stroke")
    
    if 'smoking_status' not in df.columns or 'stroke' not in df.columns:
        st.error('Necessary columns not found in the dataset')
        return
    
    try:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('smoking_status', title='Smoking Status'),
            y=alt.Y('count()', title='Count'),
            color='stroke:N'
        ).properties(
            title='Smoking Status Distribution by Stroke'
        )
        st.altair_chart(chart, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def correlation_matrix_heatmap():
    st.header("Correlation Matrix Heatmap")
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Create a heatmap using Altair
    try:
        heatmap = alt.Chart(corr_matrix.reset_index().melt(id_vars='index')).mark_rect().encode(
            x='index:O',
            y='variable:O',
            color='value:Q'
        ).properties(
            title='Correlation Matrix Heatmap'
        )
        st.altair_chart(heatmap, use_container_width=True)
    except ValueError as e:
        st.error(f"Error in generating chart: {e}")

def display_full_year_calendar():
    st.header("Full Year Calendar")

    # Get the current year
    current_year = pd.Timestamp.today().year

    # Display calendars for all months of the year
    for month in range(1, 13):
        st.subheader(calendar.month_name[month])
        month_calendar = calendar.monthcalendar(current_year, month)
        for week in month_calendar:
            st.write(week)

if __name__ == '__main__':
    main()
