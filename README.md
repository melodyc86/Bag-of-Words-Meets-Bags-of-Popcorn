# Bag-of-Words-Meets-Bags-of-Popcorn

# Project Overview
This project is a part of the Kaggle competition "Bag of Words Meets Bags of Popcorn". It focuses on sentiment analysis of movie reviews using various machine-learning techniques. The goal is to accurately predict the sentiment of a movie review (positive/negative) based on textual data with the use of machine learning. I chose to do this by utilizing PyTorch.

Link: https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview

# Data Description
The dataset comprises movie reviews from IMDB with binary sentiment classification. The training set contains X number of labeled reviews and the test set contains Y number of reviews without labels.
  
# Methodology
## Data Preprocessing
- Text Data Loading: Utilized Pandas to load the training and test datasets from CSV files.
- Data Splitting: Split the training data into training and validation sets using train_test_split from scikit-learn, ensuring a fair distribution for model training and validation.
- Vectorization: Applied CountVectorizer to convert text data into numerical vectors, a necessary step for feeding textual data into a machine learning model.
## Model Development
- Neural Network Architecture: Developed a simple feedforward neural network using PyTorch, with layers defined in the TextClassifier class. The model consists of fully connected layers and ReLU activations.
- Training Process: Trained the model over the dataset using Cross-Entropy Loss and the Adam optimizer. The training loop involves forward propagation, loss computation, backpropagation, and optimization steps.
## Performance Evaluation
- Validation Accuracy: Implemented a validation step to evaluate the model's performance on unseen data, ensuring generalization.
- Testing and Predictions: Applied the trained model to the test dataset to predict sentiments. Used softmax function to obtain probabilities and classified reviews as positive or negative based on a threshold.

# Results 
My submission achieved a score of 0.94833 or 94.83% accuracy in the competition. This indicated it was highly effective in classifying sentiments of movie reviews. 
I've also included the sentient.csv file for future analysis.

# Languages/Libraries used:
- Python
- PyTorch for model building
- Pandas and NumPy for data manipulation
- Scikit-learn for data preprocessing
- Matplotlib for data visualization
