# Recipe Time Category Prediction

**Name(s)**: Ammie Xie

**Website Link**: https://axie0927.github.io/Recipe-Category-Prediction/

### Framing the Problem

For this data science project, continuing on my investigation from my previous project and I am looking to predict the time category of a recipe based on multiple features including the number of ingredients, number of steps, calories, as well as year the recipe is submitted. Using these features, I aim to predict the time category for a recipe. The `time_category` column of the dataframe is categorized by the '60-minutes-or-less' tag. This means that all recipes that take 60 minutes or less to make are considered 'short' while recipes that take more than that are considered 'long'. 

The dataset I am using to investigate this problem is the dataset used in my previous recipes research project. This dataset is obtained from food.com containing different recipes as well as reviews by users on the website reviewing the recipes collected from 2008. The first dataset contains 83782 rows, and 12 columns. 

For this prediction, the variable I will be predicting is the 'time_category' column. This is a categorical outcome which means it is qualitative. It is important to be able to predict the time category of the recipe as it is usually helpful when individuals are looking for a recipe to know whether the recipe can be quickly replicated or it will take a long time to prepare. This allows users to quickly decide whether they want to learn this recipe or not.

In order to make this prediction, I will be using a classification model since the column we are predicting is nominal data. Furthermore this is supervised learning due to the fact that our data is labelled, we are modelling based off the 'time_category' column that I created above. As a result, this will be binary classification since there are only two outcomes 'short' and 'long'.

We will be measuring the accuracy of the model. This is because previously we found out that there are significantly more short recipes than long recipes, leading our distribution to be uneven and thus testing for accuracy would be a fair metric to test the performance of our model. 

I have selected 5 features to investigate, these features include the columns `n_steps`, `n_ingredients`, `calories`, `avg_rating` and `year`. These features with the engineering accordingly in order to maximize the performance of our model. However, these features may not all be used as using too many features may sometimes lead to overfitting. 