# Vehicle-Price-prediction
Using Logistic Regression and SVM algorithm we are trying to predict a customer will buy a vehicle or not.

final1.csv --> this is validation data or future data.
logit classification.csv --> this is historical data.

The result given by logit algo is stored in --> pred_model_by_logit.csv

The result given by svm algo is stored in --> pred_model_by_svm.csv

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Vehicle Purchase Prediction</title>
</head>
<body>

  <h1>🚗 Vehicle Purchase Prediction Using Logistic Regression and SVM</h1>

  <p>
    This project aims to predict whether a customer will purchase a vehicle based on historical data. By employing Logistic Regression and Support Vector Machine (SVM) algorithms, the model analyzes customer attributes to determine purchasing intent.
  </p>

  <h2>📌 Project Overview</h2>
  <ul>
    <li>Utilizes <strong>Logistic Regression</strong> and <strong>Support Vector Machine (SVM)</strong> algorithms.</li>
    <li>Analyzes customer data to predict vehicle purchase decisions.</li>
    <li>Compares model performance to identify the most effective approach.</li>
  </ul>

  <h2>📁 Dataset Description</h2>
  <ul>
    <li><strong>logit classification.csv</strong>: Historical customer data used for training the models.</li>
    <li><strong>final1.csv</strong>: New customer data for validation and prediction.</li>
  </ul>

  <h2>🧰 Methodology</h2>
  <ol>
    <li><strong>Data Preprocessing</strong>
      <ul>
        <li>Handling missing values and outliers.</li>
        <li>Encoding categorical variables.</li>
        <li>Feature scaling for model compatibility.</li>
      </ul>
    </li>
    <li><strong>Model Training</strong>
      <ul>
        <li>Implementing Logistic Regression and SVM algorithms.</li>
        <li>Training models on historical data.</li>
      </ul>
    </li>
    <li><strong>Model Evaluation</strong>
      <ul>
        <li>Assessing performance using metrics like accuracy, precision, recall, and F1-score.</li>
      </ul>
    </li>
    <li><strong>Prediction</strong>
      <ul>
        <li>Applying trained models to new customer data.</li>
        <li>Storing predictions in respective CSV files.</li>
      </ul>
    </li>
  </ol>

  <h2>🚀 Getting Started</h2>
  <h3>Prerequisites</h3>
  <ul>
    <li>Python 3.x installed on your system.</li>
    <li>Required Python libraries:
      <ul>
        <li>pandas</li>
        <li>numpy</li>
        <li>scikit-learn</li>
      </ul>
    </li>
  </ul>

  <h3>Installation</h3>
  <pre><code>git clone https://github.com/kunalmahadule/Vehicle-Price-prediction.git
cd Vehicle-Price-prediction
pip install pandas numpy scikit-learn</code></pre>

  <h3>Running the Models</h3>
  <ul>
    <li><strong>Logistic Regression</strong>:
      <pre><code>python Logistic_Reg_Class.py</code></pre>
    </li>
    <li><strong>Support Vector Machine (SVM)</strong>:
      <pre><code>python SVM.py</code></pre>
    </li>
  </ul>

  <h2>📂 Output Files</h2>
  <ul>
    <li><strong>pred_model_by_logit.csv</strong>: Predictions from the Logistic Regression model.</li>
    <li><strong>pred_model_by_svm.csv</strong>: Predictions from the SVM model.</li>
  </ul>

  <h2>📁 Project Structure</h2>
  <pre><code>├── Logistic_Reg_Class.py
├── SVM.py
├── final1.csv
├── logit classification.csv
├── pred_model_by_logit.csv
├── pred_model_by_svm.csv
└── README.md</code></pre>

  <h2>🤝 Contributing</h2>
  <p>Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.</p>


</body>
</html>




