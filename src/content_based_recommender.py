"""
Author: Brandon Golshirazian
Date: 12/13/2021
Description: Contains the code to do feature processing for the word association matrix
in the exploration notebook. Also provides the code for the YelpDataFeatureProcessor class
to transform the datasets code for use by ContentBaseRecommender inm model testing notebook
"""

import pandas as pd
import numpy as np
from src.feature_processing import YelpDataFeatureProcessor

class ContentBasedRecommender():
    """
    A class used to select a target user to provide recommendations to. Model will 
    transform data, fit the data (for the specified user), then predict the top
    recommendations for the user.

    Attributes
    ----------
    business_data : DataFrame
        data frame of business data from yelp dataset
    user_data : DataFrame
        data frame of user data from yelp dataset
    review_data : DataFrame
        data frame of review data from yelp dataset
    target_user: str
        string of user's unique id that we're trying to provide recommendations for
    recommendations: ndarray
        numpy array of restaurant indices to be recommended after using fit()
    _restaurant_rec_scores: ndarray
        numpy array of restaurants where columns are restaurant indices (according to business_data)
        and calculated recommendation scores
    _transform: YelpDataFeatureProcessor
        class that contains the methods to transform yelp datset into feature
        matrices/feature vectors for model building/usage
    _user_weights: ndarray
        contains the user calculated weights after processing reviews, and other user data

    Methods
    -------
    fit(target_user)
        train the model using a specified target_user (id) from the user_data set. This function
        transforms all the necessary data and performs calculations using that trained info
    
    predict()
        return list of recommendations for the user sorte by top rated businesses for the user


    """
    def __init__(self, business_data, user_data, review_data):
        self.business_data = business_data
        self.user_data = user_data
        self.review_data = review_data
        self._user_weights = None
                
    
    def fit(self, target_user):
        """
        fit takes in a target_user str and transforms all relevant, business & review data to train
        the model. Recommended restaurant indices will be in self.recommendations. To look at 
        specific scores refer to self._restaurant_rec_scores

        :param target_user: str user_id from user_data

        """
        self.target_user = target_user
        self._transform = YelpDataFeatureProcessor(
            target_user,
            self.business_data,
            self.user_data,
            self.review_data
        )

        # get restaurant features & ratings
        visited_restaurant_features, visited_rest_ratings = \
            self._transform.transform_visited_restaurants()

        # to calculate weights for user and see preferences
        self._user_weights = np.dot(visited_rest_ratings, visited_restaurant_features)

        # one hot encode all other businesses to provide recommendations
        all_restaurant_features = self._transform.transform_new_restaurants()        
        
        # calculate score for each business using self._user_weights
        restaurant_rec_score = np.zeros(all_restaurant_features.shape[0])
        for i in range(all_restaurant_features.shape[0]):
            restaurant_rec_score[i] = np.sum(all_restaurant_features[i] * self._user_weights) / np.sum(self._user_weights)        

        # combine restaurant indices with recommendation score and sort by score descending
        restaurant_id_and_rec_score = np.vstack(
            (restaurant_rec_score, np.arange(all_restaurant_features.shape[0]))
        ).T

        self._restaurant_rec_scores = restaurant_id_and_rec_score[
            np.argsort(restaurant_id_and_rec_score[:, 0])[::-1]
        ]

        self.recommendations = self._restaurant_rec_scores[:, 1].astype(int)
        
        
    def predict(self):
        """
        predict will return the busienss_data DataFrame sorted by highly recommended businesses
        at the top. 

        :return: DataFrame of businesses sorted by recommendation score descending
        """
        return self.business_data.iloc[self.recommendations][["name", "stars", "categories"]]

