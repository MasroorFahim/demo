# House price analysis by Artificial Neural Networks 
hello, so here we gonna take a dataset of housingdataset and see what we can do with the dataset.



We will



1. Import the necessary packages.
2. Visualise the columns in the data and check the variance threshold to check whether there is any non zero columns in the dataset.
3. Data Preprocessing and cleaning and it will be generally be donw with some plotting and perason correlation techniques
4. Training and testing split
5. Scaling/Minmax Scaling and normalization
6. Building model (ANN)
7. Five Metrics are used and they are 1.R2_score,2.Mean_absolute_error,3.mean_squared_error,4.Mean_squared_log_error and 5. Explained_variance_score and Score them accordingly.
8. Predictive System with the actual price.



1. We imported wide range of functionalities necessary for data manipulation, visualization, preprocessing, and building neural network models using Keras.
 
Dataset Overview: Initially, the dataset consisted of 10 features. After data cleaning, handling missing values, using a variance threshold, and further processing, the dataset contained 20,433 rows.
  
2. We visualised the columns and used Variance thresholding, which is a technique used in feature selection to remove features (columns) from a dataset that have low variance. It's commonly applied before model training to eliminate features that essentially have almost the same value across all samples in the dataset, as these features might not contribute much to the model's predictive power.
In Python's scikit-learn library, you can find the VarianceThreshold class in the feature_selection module, which helps with this task.

3.Exploratory Data Analysis (EDA):

Box Plotting: Visual representation of the distribution of data points, especially for identifying outliers or the spread of data within each feature.

Pearson Correlation and PCA: 
Pearson correlation is a statistical measure that quantifies the linear relationship between two continuous variables. It assesses how strongly the variables are related on a scale from -1 to 1.

A correlation coefficient of 1 indicates a perfect positive linear relationship: As one variable increases, the other also increases proportionally.
A coefficient of -1 indicates a perfect negative linear relationship: As one variable increases, the other decreases proportionally.
A coefficient close to 0 indicates no linear relationship between the variables

4.Data Splitting:

Train-Test Split: The dataset was split into a training set (80%) and a test set (20%) with a specified random state (123) to maintain reproducibility as well as stratified to make it random.

5.Feature Scaling:

Min-Max Scaling: Scaling the features to a range between 0 and 1 to ensure uniformity and prevent certain features from dominating the model due to their scale.

6.Model Building (ANN):

Artificial Neural Network (ANN): A type of machine learning model inspired by the human brain's neural structure, composed of layers of interconnected nodes. You've built and trained an ANN using the processed data.

ANN stands for Artificial Neural Network. It's a computational model inspired by the structure and function of the human brain's neural networks. An ANN is composed of interconnected nodes, often organized in layers, which process information and make predictions or classifications.

Key components of an Artificial Neural Network:

Neurons (Nodes): These are the basic units of computation. Each neuron takes inputs, performs a computation, and produces an output.

Layers: Neurons are organized into layers within the network. The most common layers include:

Input Layer: Receives the initial input data.
Hidden Layers: Intermediate layers between the input and output layers where computations occur.
Output Layer: Produces the final output, which could be a prediction, classification, or any desired outcome.
Connections (Weights and Biases): Neurons are connected to neurons in adjacent layers. Each connection has an associated weight that represents its strength. Additionally, neurons often have an associated bias, influencing the output.

Activation Function: Each neuron typically applies an activation function to its weighted inputs to introduce non-linearities into the network, enabling it to learn complex patterns in the data.

Training an ANN involves the following steps:

Forward Propagation: During training, data is fed forward through the network. Each neuron's input is calculated by combining the weighted sum of inputs from the previous layer with its bias, which is then transformed by the activation function to produce the neuron's output.

Backpropagation: After the forward pass, the network's output is compared to the desired output (in supervised learning tasks). The difference (error) is calculated, and this error is propagated backward through the network to adjust the weights and biases using optimization algorithms (like gradient descent) to minimize the error.

Learning: The process of adjusting weights and biases continues iteratively (over multiple epochs) until the network's predictions or classifications align closely with the actual outputs in the training data.

ANNs have gained popularity due to their ability to learn complex patterns from data, adapt to non-linear relationships, and perform well in various tasks such as classification, regression, and pattern recognition. They are the fundamental building blocks for more advanced architectures like Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs).


7.. Evaluation Metrics:

R-squared (R²) Score: Represents the proportion of variance in the dependent variable (target) that is predictable from the independent variables. Higher values indicate a better fit.
Mean Absolute Error (MAE): The average of the absolute differences between predicted and actual values, providing the magnitude of errors.
Mean Squared Error (MSE): The average of the squared differences between predicted and actual values, amplifying larger errors.
Mean Squared Log Error (MSLE): Similar to MSE, but calculated on the logarithm of the predicted and actual values.
Explained Variance Score: Indicates the proportion of variance captured by the model.

R-squared (R²) Score:

R-squared:  measures the proportion of variance in the dependent variable (target) that is predictable from the independent variables (features).
It ranges from 0 to 1, where 1 indicates that the model perfectly predicts the target variable and 0 indicates that the model doesn't explain any variance in the target variable beyond the mean.


MAE: calculates the average absolute differences between predicted values and actual values.
It provides a measure of the average magnitude of errors without considering their direction.
It is calculated as the average of absolute differences between predicted and actual values.
Mean Squared Error (MSE):

MSE : measures the average of the squared differences between predicted values and actual values.
It amplifies larger errors due to squaring the differences.
It provides a more detailed understanding of the model's performance by penalizing larger errors more significantly.
It is calculated as the average of squared differences between predicted and actual values.
Mean Squared Logarithmic Error (MSLE):

MSLE:  is similar to MSE but calculates the logarithm of the predicted and actual values before computing the squared differences.
It is particularly useful when the target values have exponential trends.
It measures the ratio between the true and predicted values, penalizing underestimates more than overestimates.
Explained Variance Score:

Explained Variance Score: quantifies the proportion of variance in the target variable that the model captures.
It indicates how well the model accounts for variability in the dataset.
The best possible score is 1.0, indicating perfect prediction.
Each of these metrics serves a different purpose in evaluating the performance of machine learning models. While R-squared and Explained Variance Score assess the overall model fit, MAE, MSE, and MSLE focus on the accuracy and magnitude of errors between predicted and actual values. The selection of these metrics depends on the specific problem context and the trade-offs you want to consider in evaluating your model's performance.

7..Graphical Visualization:

Plotting Training and Validation Graphs: Visual representations of model performance during training and validation phases.
Final Prediction:

The trained model was used to make predictions on new data.
It seems like the dataset we took by combining combining data preprocessing, model building, evaluation, and visualization. The conclusion suggests that while an ANN was used for educational purposes and was only used to demonstate how it is used, other models like regression or logistic regression might be more suitable for this dataset. And last of all, it is just a demonstation of how we can use ANN in any dataset, and here we have given techniques to fit the dataset nicely.

For pricing, the complexity and time taken for such a project depend on various factors like the dataset size, computational requirements, and the depth of analysis. It might be wise to consider these factors when estimating the project's cost or time required for completion




