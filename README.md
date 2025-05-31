# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RITZ LONGJAM

*INTERN ID*: 

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

#TASK DESCRIPTION:Building and Visualizing a Decision Tree Model Using Scikit-Learn

The given task was to build and visualize a Decision Tree model using the Scikit-learn library in Python to classify or predict outcomes on a chosen dataset. The goal was to demonstrate an understanding of decision tree classification, data preprocessing, model building, evaluation, and visualization, all compiled into a structured and interpretable notebook. The dataset chosen for this task was the Titanic dataset, which is widely used for classification tasks in machine learning and provides a real-world scenario of predicting survival based on various passenger features.

Tools and Technologies Used:

The programming language used for this task was Python, and the entire implementation was carried out on Google Colab, which is an online cloud-based platform that supports Python notebooks and allows easy access to GPU/TPU, collaboration, and sharing.

Several essential Python libraries were used in this notebook:

•	Pandas: For loading and manipulating the dataset.

•	NumPy: For numerical operations and array handling.

•	Seaborn: For data loading and visualization (e.g., confusion matrix).

•	Matplotlib: For plotting graphs and trees.

•	Scikit-learn (sklearn): The core machine learning library used to split the dataset, train the Decision Tree Classifier, evaluate model performance, and visualize the decision tree.

Modules includes:

1.	Dataset Loading and Cleaning:
The Titanic dataset was loaded directly using Seaborn's built-in load_dataset() function. It contains passenger information such as class, age, fare, gender, and whether the passenger survived or not. The dataset was preprocessed by selecting relevant features and removing rows with missing values to ensure clean and consistent data for training.

3.	Feature Encoding:
Categorical variables like sex and embarked were converted into numerical format using mapping. This is a crucial step for any machine learning model since models can only interpret numerical values.

5.	Splitting the Dataset:
The dataset was split into training and testing sets using an 80-20 split ratio via train_test_split. This allows us to train the model on one portion and evaluate its performance on unseen data.

7.	Model Building and Training:
A Decision Tree Classifier from sklearn.tree was used. The tree depth was limited to 4 using max_depth=4 to prevent overfitting and ensure that the model generalizes well. The model was then trained using the .fit() method.

9.	Model Visualization:
One of the key deliverables was visualizing the decision tree. This was done using plot_tree(), which clearly displayed the decision paths, features used at each node, Gini impurity, and class predictions. The visualization makes the decision-making process of the model interpretable, which is one of the strengths of decision trees.

11.	Model Evaluation:
Predictions were made on the test set using the trained model. The model's performance was evaluated using a classification report showing precision, recall, f1-score, and accuracy. Additionally, a confusion matrix was plotted using Seaborn's heatmap to better understand the distribution of true positives, false positives, true negatives, and false negatives.

Applications

Decision trees are widely used in various domains for their interpretability and simplicity. The Titanic survival prediction is just a classic example of a binary classification problem. Real-world applications include:

•	Medical diagnosis (e.g., predicting disease based on symptoms)

•	Loan approval systems

•	Customer churn prediction

•	Fraud detection

•	Risk assessment models in insurance

*OUTPUT*

<img width="620" alt="Image" src="https://github.com/user-attachments/assets/bbc43f39-89b8-4fc6-aa5f-044d82ee5c04" />

<img width="367" alt="Image" src="https://github.com/user-attachments/assets/5695c609-d9a7-4f1e-874d-d2591cdc485f" />

<img width="405" alt="Image" src="https://github.com/user-attachments/assets/b6b0a74e-28c8-42cf-8c72-6e7f89a6b51e" />


