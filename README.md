# SMS Spam Classifier — NLP Project

## Overview

This project implements a **spam filtering model for SMS messages** using Natural Language Processing (NLP) and classic machine learning techniques. It demonstrates the process of cleaning raw SMS data, extracting features, building and evaluating multiple classification models, and selecting the most effective one for spam detection.

## Features

- **Data Cleaning & Preprocessing:** Removal of noise, stopwords, punctuations, and lower-casing.
- **Exploratory Data Analysis (EDA):** Visualizations and statistics on spam and ham (not spam) distribution.
- **Feature Engineering:** Utilizes techniques like Bag-of-Words and TF-IDF for text vectorization.
- **Multiple Classifiers:** Models built using algorithms such as Multinomial Naive Bayes, Support Vector Machine (SVM), and Random Forest.
- **Performance Evaluation:** Compares models using metrics like accuracy, precision, recall, F1-score, and confusion matrix.

## Tech Stack

| Technology | Purpose                        |
|------------|-------------------------------|
| Python     | Programming Language           |
| Jupyter    | Notebook environment           |
| scikit-learn | ML algorithms, model evaluation |
| pandas     | Data handling                  |
| NLTK       | Natural Language Processing    |
| matplotlib / seaborn | Data visualization   |

## Getting Started

### Prerequisites

- Python (3.x recommended)
- Jupyter Notebook or JupyterLab
- Install dependencies with:
    ```
    pip install pandas scikit-learn nltk matplotlib seaborn
    ```

### Dataset

- The project uses the classic [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
- Download and place it in your working directory as `spam.csv` or as specified in the notebook.

### Running the Notebook

1. Clone or download this repository:
    ```
    git clone https://github.com/devansh19020/sms-spam-nlp.git
    ```
2. Open the notebook in Jupyter:
    ```
    cd sms-spam-nlp
    jupyter notebook sms-spam-nlp.ipynb
    ```
3. Follow the steps in the notebook to run each cell sequentially.

## Folder Structure

<pre>
sms-spam-nlp/
├── sms-spam-nlp.ipynb 
├── README.md
</pre>


## Usage

- Run all cells of the notebook to train spam detection models and evaluate their performance.
- The final cells demonstrate predictions on sample messages and summarize model results.

## Results

- After exploring several algorithms, the notebook reports the classification performance.
- Typical results for Naive Bayes on this dataset: **Accuracy ~98%**; strong precision/recall for spam detection.
- Check the notebook output cells for detailed confusion matrices and metric tables.

## Contributing

Ideas, bug reports, and contributions are welcome! Fork the repository, make your changes, and open a pull request.

## License

This project is free to use for educational and research purposes.

## Acknowledgements

- Built as a basic NLP mini-project while learning spam filtering and text classification.
- Dataset: [UCI Machine Learning Repository - SMS Spam Collection][1]

---

[1]: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
