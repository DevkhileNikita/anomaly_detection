import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import smtplib
import mplcursors
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import EmailMessage
from sklearn.model_selection import GridSearchCV


# Function to perform anomaly detection

def perform_anomaly_detection(data, selected_y_column, start_date, end_date):

    # Filter data based on the selected date range

    selected_data = data[(data['day'] >= pd.to_datetime(start_date)) & (data['day'] <= pd.to_datetime(end_date))]

    # Check if there is enough data for anomaly detection

    if selected_data.empty:

        st.warning("No data available for the selected date range.")

        return pd.DataFrame()

    # Perform anomaly detection using Isolation Forest

    iso_forest = IsolationForest(contamination=0.05)

    iso_forest.fit(selected_data[[selected_y_column]])

    selected_data['Outlier'] = np.where(iso_forest.predict(selected_data[[selected_y_column]]) == -1, 1, 0)

    return selected_data

    # Check if anomalies are detected and trigger email notification

    if num_anomalies > 0:

        send_email_notification()

    return selected_data

# Function to generate Line Chart with hover tooltip

    def generate_line_chart(data, selected_x_column, selected_y_column):

        fig, ax = plt.subplots(figsize=(10, 6))

    line = ax.plot(data[selected_x_column], data[selected_y_column], marker='o')

    plt.xlabel(selected_x_column)

    plt.ylabel(selected_y_column)

    plt.title(f'Line Chart of {selected_y_column} over {selected_x_column}')

    # Add hover tooltip to the line chart

    mplcursors.cursor(line)

    return fig

# Function to send email notification

def send_email_notification():

    email_sender = 'nikita.b@prodapr.com'  # Replace with your email

    email_receiver = 'kamakshi.b@prodapt.com'   # Replace with the recipient's email

    smtp_server = 'smtpgw01.mediassistindia.com'   # Replace with your SMTP server address

    smtp_port = 587                                 # Replace with your SMTP server port number

    # Set up the SMTP server

    server = smtplib.SMTP(smtp_server, smtp_port)

    server.starttls()

    server.login(email_sender, 'Aug851996@')  # Replace with your email password/API key

    # Compose the email content

    subject = 'Anomaly Detected!'

    body = 'An anomaly has been detected in the data.'

    message = MIMEMultipart()

    message['From'] = email_sender

    message['To'] = email_receiver

    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    # Send the email

    server.sendmail(email_sender, email_receiver, message.as_string())

    server.quit()

# Function to filter data based on the selected date range

def filter_data(data, selected_x_column, selected_y_column, start_date, end_date):

    selected_data = data[(data['day'] >= pd.to_datetime(start_date)) & (data['day'] <= pd.to_datetime(end_date))]

    # Perform anomaly detection using Isolation Forest

    iso_forest = IsolationForest(contamination=0.05)

    iso_forest.fit(selected_data[[selected_y_column]])

    selected_data['Outlier'] = iso_forest.predict(selected_data[[selected_y_column]])

    return selected_data

# Function to generate Outlier Graph (Scatter Plot) with colors for Anomaly Count

def generate_outlier_scatter_plot(data, selected_x_column, selected_y_column, start_date, end_date):

    # Filter data based on the selected date range

    selected_data = data[(data['day'] >= pd.to_datetime(start_date)) & (data['day'] <= pd.to_datetime(end_date))]

    # Initialize list to store anomaly count for each date

    anomaly_count = []

# Function to generate Outlier Line Graph for Anomaly Count

def generate_outlier_line_graph(data, selected_x_column, selected_y_column, start_date, end_date):

    # Filter data based on the selected date range

    selected_data = data[(data['day'] >= pd.to_datetime(start_date)) & (data['day'] <= pd.to_datetime(end_date))]

    # Initialize list to store anomaly count for each date

    anomaly_count = []

    # Iterate through each date and calculate anomaly count

    for date in pd.date_range(start=start_date, end=end_date, freq='D'):

        selected_date_data = selected_data[selected_data['day'] == date]

        if not selected_date_data.empty:

            iso_forest = IsolationForest(contamination=0.05)

            iso_forest.fit(selected_date_data[[selected_y_column]])

            selected_date_data['Outlier'] = iso_forest.predict(selected_date_data[[selected_y_column]])

            num_anomalies = (selected_date_data['Outlier'] == -1).sum()

            anomaly_count.append(num_anomalies)

        else:

            # No data available for the current date

            anomaly_count.append(0)

    # Generate the outlier line graph

    fig = plt.figure(figsize=(10, 6))

    plt.plot(pd.date_range(start=start_date, end=end_date, freq='D'), anomaly_count, marker='o')

    plt.xlabel('Date')

    plt.ylabel('Anomaly Count')

    plt.title('Anomaly Count for each Date')

    plt.xticks(rotation=45)

    return fig

# Function to generate Bar Chart

def generate_bar_chart(data, selected_x_column, selected_y_column):

    fig = plt.figure(figsize=(10, 6))

    plt.bar(data[selected_x_column], data[selected_y_column])

    plt.xlabel(selected_x_column)

    plt.ylabel(selected_y_column)

    plt.title(f'{selected_y_column} vs {selected_x_column}')

    return fig

# Function to generate Histogram

def generate_histogram(data, selected_y_column):

    fig = plt.figure(figsize=(10, 6))

    plt.hist(data[selected_y_column], bins=20, edgecolor='k')

    plt.xlabel(selected_y_column)

    plt.ylabel('Frequency')

    plt.title(f'Histogram of {selected_y_column}')

    return fig

# Function to generate Line Chart

def generate_line_chart(data, selected_x_column, selected_y_column):

    fig = plt.figure(figsize=(10, 6))

    plt.plot(data[selected_x_column], data[selected_y_column], marker='o')

    plt.xlabel(selected_x_column)

    plt.ylabel(selected_y_column)

    plt.title(f'Line Chart of {selected_y_column} over {selected_x_column}')

    return fig

    # Function to generate Box Plot
def generate_box_plot(data, selected_x_column, selected_y_column):
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[selected_x_column], y=data[selected_y_column], color='skyblue')
    plt.xlabel(selected_x_column)
    plt.ylabel(selected_y_column)
    plt.title(f'Box Plot of {selected_y_column} across {selected_x_column}')
    plt.xticks(rotation=45)
    return fig

# Function to generate Outlier Scatter Plot for Anomaly Count

def generate_outlier_scatter_plot(data, selected_x_column, selected_y_column, start_date, end_date):

    # Filter data based on the selected date range

    selected_data = data[(data['day'] >= pd.to_datetime(start_date)) & (data['day'] <= pd.to_datetime(end_date))]

    # Initialize list to store anomaly count for each date

    anomaly_count = []

def main():

    st.title('Anomaly Detection Dashboard')

    # File Upload and Data Processing
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    data = pd.DataFrame()  # Initialize data to an empty DataFrame
    selected_data = pd.DataFrame()
    selected_x_column = None
    selected_y_column = None
    start_date = None
    end_date = None

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['day'] = pd.to_datetime(data['day'], format='%d-%m-%Y')  # Convert 'day' column to datetime

        st.write("## Dataset Information")
        st.write("Number of Rows:", data.shape[0])
        st.write("Number of Columns:", data.shape[1])
        st.write("## Your Dataset")
        st.write(data)

        # Choose X and Y columns for visualizations
        selected_x_column = st.sidebar.selectbox("Select X-Axis Column:", data.columns)
        selected_y_column = st.sidebar.selectbox("Select Y-Axis Column:", data.columns)

        # Anomaly Detection
        st.write("## Anomaly Detection")

        # Date Range Selection
        st.write("Select Date Range for Anomaly Detection:")
        start_date, end_date = st.sidebar.date_input("Date Range:", [data['day'].min(), data['day'].max()])


        # Call perform_anomaly_detection with the correct arguments
        selected_data = perform_anomaly_detection(data, selected_y_column, start_date, end_date)

        # Check if there are any outliers in the selected_data DataFrame
        if 'Outlier' not in selected_data:
            st.warning("No outliers found for the selected date range.")
            return

        # Calculate the count of anomalies
        num_anomalies = (selected_data['Outlier'] == 1).sum()

        # Calculate the percentage of anomalies
        total_data_points = len(selected_data)
        percentage_anomalies = (num_anomalies / total_data_points) * 100

        # Display the count and percentage of anomalies
        st.write(f"Total Number of Anomalies: {num_anomalies}")
        st.write(f"Percentage of Anomalies: {percentage_anomalies:.2f}%")

        # Show the filtered data as a table (include only rows with anomalies)
        st.write("## Filtered Data (Anomalies Only)")
        anomaly_data = selected_data[selected_data['Outlier'] == 1]
        st.write(anomaly_data)

        # Define the range of contamination values to search through
        contamination_values = [0.01, 0.05, 0.1, 0.15, 0.2]

        # Perform grid search with cross-validation
        iso_forest_model = IsolationForest(random_state=42)
        grid_search = GridSearchCV(iso_forest_model, param_grid={'contamination': contamination_values}, cv=5, scoring='f1')
        grid_search.fit(selected_data[[selected_y_column]])

        # Get the best contamination value
        best_contamination = grid_search.best_params_['contamination']

        # Print the best contamination value
        st.write(f"Best contamination value: {best_contamination}")

    if selected_x_column is not None and selected_y_column is not None:

        # Generate Bar Chart

        st.write("## Bar Chart")

        bar_chart = generate_bar_chart(selected_data, selected_x_column, selected_y_column)

        st.pyplot(bar_chart)

        # Generate Histogram

        st.write("## Histogram")

        histogram_chart = generate_histogram(selected_data, selected_y_column)

        st.pyplot(histogram_chart)

        # Generate Box Plot
        st.write("## Box Plot for Outlier Detection")
        box_plot = generate_box_plot(selected_data, selected_x_column, selected_y_column)
        st.pyplot(box_plot)

        # Generate Line Chart

        st.write("## Line Chart")

        line_chart = generate_line_chart(selected_data, selected_x_column, selected_y_column)

        st.pyplot(line_chart)

        # Generate Outlier Scatter Plot for Anomaly Count

        st.write("## Anomaly Detection (Line Graph)")

        outlier_line_graph = generate_outlier_line_graph(data, selected_x_column, selected_y_column, start_date, end_date)

        st.pyplot(outlier_line_graph)



if __name__ == '__main__':

    main()