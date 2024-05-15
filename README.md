# SMS Spam Detection

This repository contains a machine learning project focused on detecting SMS spam messages. The project includes data preprocessing, text transformation, visualization, feature extraction, model training, and evaluation steps. Several machine learning algorithms are implemented to classify SMS messages as spam or ham (not spam).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The purpose of this project is to build a machine learning model that can classify SMS messages as spam or ham. The project uses various natural language processing (NLP) techniques and machine learning algorithms to achieve this.

## Features

- Data preprocessing: Cleaning and preparing the SMS spam dataset for modeling.
- Text transformation: Converting text data into numerical features using techniques like tokenization, removing stop words, and stemming.
- Data visualization: Visualizing the distribution of features and their relationship with spam and ham messages.
- Model training: Implementing various algorithms including Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, and more to train a spam detection model.
- Model evaluation: Assessing the performance of each model using accuracy and precision metrics.

## Dataset

The dataset used in this project is the [SMS Spam Collection Data Set](https://www.kaggle.com/uciml/sms-spam-collection-dataset) from Kaggle. It contains labeled SMS messages as spam or ham.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Aanchal2707/sms-spam-detection.git
    cd sms-spam-detection
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the dataset file (`spam.csv`) in the project directory. If not, download it from the provided link and place it there.

2. Run the script to execute data preprocessing, text transformation, visualization, model training, and evaluation:
    ```bash
    python spam_detection.py
    ```

## Data Visualization

The project includes various data visualization steps to understand the distribution of features and their relationship with spam and ham messages. Key visualizations include:

- Histograms showing the distribution of the number of characters, words, and sentences in spam and ham messages.
- Pair plots to visualize relationships between features.
- Heatmaps to show correlations between features.

## Model Training and Evaluation

The script `spam_detection.py` performs the following steps:

1. Loads and preprocesses the data.
2. Transforms the text data using techniques like tokenization, removing stop words, and stemming.
3. Visualizes the data to understand feature distributions and relationships.
4. Extracts features using techniques like Count Vectorizer and TF-IDF Vectorizer.
5. Splits the data into training and testing sets.
6. Trains several machine learning models including Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree, Random Forest, and various ensemble models.
7. Evaluates each model using accuracy and precision metrics.

## Results

The results of the model evaluations are summarized in a DataFrame showing the accuracy and precision of each model:

- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- Bagging Classifier
- Extra Trees Classifier
- HistGradient Boosting Classifier
- XGBoost Classifier
- XGBoost RF Classifier

```python
performance_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
}).sort_values(['Precision', 'Accuracy'], ascending=False)
print(performance_df)
```

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to reach out if you have any questions or suggestions! Happy coding!

---

**Note:** Replace `Aanchal2707` with your actual GitHub username in the clone URL.
