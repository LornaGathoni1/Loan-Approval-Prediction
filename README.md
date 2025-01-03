﻿# Loan-Approval-Prediction
 # Loan Approval Prediction System

This project is a **Loan Approval Prediction System** built using a Machine Learning model to predict loan approval based on user inputs. The app is deployed using **Streamlit** and is designed to provide an intuitive interface for users to input their data and receive predictions.

## Features
- **User-friendly Interface**: Easy-to-use form for inputting data like gender, income, and loan details.
- **Real-time Predictions**: Displays loan approval status instantly upon form submission.
- **Scalable Model**: Trained on a robust dataset to predict loan approvals with high accuracy.

## Dataset
The model is trained on a **Loan Prediction Dataset**, which includes the following features:
- **Gender**: Male or Female.
- **Married**: Yes or No.
- **Dependents**: Number of dependents.
- **Education**: Graduate or Not Graduate.
- **Self_Employed**: Yes or No.
- **ApplicantIncome**: Income of the applicant.
- **CoapplicantIncome**: Income of the co-applicant (if any).
- **LoanAmount**: Loan amount in thousands.
- **Loan_Amount_Term**: Duration of the loan in months.
- **Credit_History**: 1 for a good credit history, 0 for a bad one.
- **Property_Area**: Urban, Semi-Urban, or Rural.

The dataset is preprocessed to handle missing values, encode categorical variables, and normalize numerical features.

## How It Works
1. Input raw data (e.g., income, property area, loan details) into the form.
2. Click the **Submit** button to process your request.
3. Receive a loan status prediction: **Approved** or **Not Approved**, with highlights for clarity.

## Technologies Used
- **Machine Learning**: Trained using a Logistic Regression model.
- **Streamlit**: For deploying an interactive web application.
- **Python**: Core programming language.
- **GitHub**: Version control and collaboration.

## Screenshots
### User Input Form
![image](https://github.com/user-attachments/assets/39915a70-f540-45f7-aa76-e1014b2458eb)


### Prediction Results
![image](https://github.com/user-attachments/assets/e70b49ee-6067-48c4-82d7-b6f663f721f0)



## How to Run the App Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/LornaGathoni1/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction

    Install the required packages:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run app.py

License

This project is open-source and available under the MIT License.
