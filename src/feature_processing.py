"""
Author: Brandon Golshirazian
Date: 12/13/2021
Description: Contains the code to do feature processing for the word association matrix
in the exploration notebook. Also provides the code for the YelpDataFeatureProcessor class
to transform the datasets code for use by ContentBaseRecommender inm model testing notebook
"""
import pandas as pd
import numpy as np

######################################
### Word Association Matrix Code   ###
######################################

def word_matrix_to_sorted_word_dict(row, col, matrix):
    """
    word_matrix_to_word_dict takes a word_association_matrix, and converts it to a dictionary,
    where keys are tuples of indexes in matrix, and the resulting value are the occurrences

    :param row: rows of the word_association_matrix
    :param col: cols of the word_association_matrix
    :param matrix: the word_association_matrix to be passed in
    :return: dict where keys are tuples of indexes in the matrix (corresponding to some word)
             and values are occurrences e.g { (100, 201): 213, ... }
    """
    associations = {}
    for i in range(row):
        for j in range(col):
            associations[(i, j)] = matrix[i, j]

    return dict(sorted(associations.items(), reverse=True, key=lambda x: x[1]))

def index_key_to_word_key(categories, word_dict):
    """
    index_key_to_word returns a list of dictionary items where the tuple keys associated with
    a word_association_matrix index are turned into words e.g { (100, 201): 213, ... } --> [("Church", "Volunteering"): 213, ... }]

    :categories param: a numpy array/pandas series where each index corresponds to its spot in the
    word_association_matrix
    :word_dict: a dictionary from word_matrix_to_sorted_word_dict of the format { (100, 201): 213, ... }
    :return: list of dictionary items with tuple keys converted to tuple of words
    """

    return list(map(lambda x: (categories[x[0][0]], categories[x[0][1]], x[1]), word_dict.items()))

def remove_duplicates(category_associations):
    """
    remove_duplicates returns a list of dictionary items (from the word_matrix_to_sorted_word_dict function)
    with the tuple keys that have the same indexes/words removed

    :param category_associations: list in the form of  [(("Chinese", "Dessert"): 213), ... ]
    :return: duplicates removed from list of tuples where inner tuple has 2 of the same words
    e.g (("Chinese", "Chinese"): 35) would be removed
    """
    return filter(lambda x: x[0] != x[1], category_associations)

def get_top_word_associations(word_association_frame, target_category):
    """
    get_top_word_associations takes in a frame with 2 columns of words associated with each other,
    the returns the rows that have a specified target_category

    :param word_association_framem: pandas DataFrame with 2 columns of categories & 1 of the occurrences
    :param target_category: the string to search both columns for
    :return: pandas DataFrame filtered by target_category sorted by occurrences
    """

    return word_association_frame[
            (word_association_frame["first_cat"] == target_category) | (word_association_frame["second_cat"] == target_category)
    ].sort_values(by="# of assoc", ascending=False)

################################################################
### Content Based Information Filtering - Feature Processing ###
################################################################

class YelpDataFeatureProcessor():

    """
    A class used to transform data from the yelp dataset. Formats data into 
    one hot encoded style feature matrices and vector. Ultimately for usage
    with numpy's linear algebra math functions.

    Attributes
    ----------
    business_data : DataFrame
        data frame of business data from yelp dataset
    user_data : DataFrame
        data frame of user data from yelp dataset
    review_data : DataFrame
        data frame of review data from yelp dataset
    target_user_id: str
        string of user's unique id that we're trying to provide recommendations for
    categories: Series
        Series of all restaurant categories accessible by index
    word_location: dict
        Dictionary where keys are a category in *categories* and value gives the location in *categories*

    Methods
    -------
    transform_visited_restaurants()
        Finds all restaurants that target_user_id has visited. This returns 2 things as a tuple:
        1) Transforms visited restaurants categories into a one hot encoded feature matrix where
        columns correspond to categories and rows are different restaurants
        2) Transforms those same visited restaurants into vector of restaurant ratings
    
    transform_new_restaurants()
        Goes through entire DataFrame of self.business_data and transforms all restaurants
        into a feature matrix. Will be used to score recommendations based on user weights
        for the top recommendations.

    """

    def __init__(self, target_user_id, business_data, user_data, review_data):
        self.business_data = business_data
        self.user_data = user_data
        self.review_data = review_data
        
        self.target_user_id = target_user_id

        self.categories = self.business_data\
                        .explode("categories")["categories"]\
                        .str\
                        .strip()\
                        .unique()
        
        print(self.categories)
        
        self.word_location = {self.categories[i]: i for i in range(len(self.categories))}

    def transform_visited_restaurants(self):
        """
        transform_visited_restaurants takes the target_user_id and extract the features
        for the restaurants he/she/they visited. Also generates matrix of ratings
        corresponding to each of those businesses

        :return: tuple of (ndarray, ndarray) visited restaurant features & user ratings for each restaurant
        """
            
        # get all reviews of a particular user
        user_reviews = self.review_data[self.review_data["user_id"] == self.target_user_id]
        
        
        # grab the business_id column and sort those values
        visited_restaurant_ids = user_reviews["business_id"]\
            .sort_values()\
            .reset_index(drop=True)

        # turn restaurant categories into features
        visited_restaurant_features = self.__get_visited_restaurant_features(
            visited_restaurant_ids
        ).to_numpy()[:, 1:]
        
        # turn restaurant ratings into 1d vector
        visited_restaurant_ratings = self.__get_visited_restaurants_ratings(
            user_reviews,
            visited_restaurant_ids
        ).to_numpy()
        
        visited_restaurant_ratings = visited_restaurant_ratings.reshape(
            1,
            visited_restaurant_ratings.shape[0]
        )

        return (visited_restaurant_features, visited_restaurant_ratings)
        

    def __get_visited_restaurant_features(self, visited_restaurant_ids):
        """
        __get_visited_restaurant_features takes in a Series of restaurants target_user_id
        visited, and turns those into a feature matrix

        :param visited_restaurant_ids: Series of str based business_id's from yelp_business_data
        :return feature_matrix: ndarray of n businesses x d unique business categories
        """
        visited_restaurant_data = self.business_data.loc[
            self.business_data["business_id"].isin(visited_restaurant_ids)
        ].sort_values(by="business_id")
        
        business_category_texts = visited_restaurant_data["categories"]

        feature_matrix = self.__business_cat_to_feature_matrix(
            visited_restaurant_ids,
            business_category_texts
        )

        return feature_matrix


    def __business_cat_to_feature_matrix(self, restaurant_ids, all_restaurant_categories):
        """
        __business_cat_to_feature_matrix accepts Series of restaurant_ids and a Series of those
        restaurants' categories. Does one hot encoding on new ndarray n businesses x d categories
        and returns that

        :param restaurant_ids: Series of str *business_id's*
        :param all_restaurant_categories: Series of sets containing business categories
        :return feature_matrix: ndarray of n businesses x d business categories
        """

        feature_matrix = pd.DataFrame(columns=self.categories)
        feature_matrix.insert(0, "business_id", restaurant_ids)
        feature_matrix = feature_matrix.reset_index(drop=True).fillna(0)

        restaurant_list_length = all_restaurant_categories.shape[0]

        for restaurant_category_list, i in zip(all_restaurant_categories, range(restaurant_list_length)):
            for category in restaurant_category_list:
                feature_matrix.iat[i, self.word_location[category] + 1] = 1
        
        return feature_matrix




    def __get_visited_restaurants_ratings(self, user_reviews, visited_restaurant_ids):
        """
        __get_visited_restaurants_ratings takes DataFrame of user reviews, and Series of
        visited_restaurant_ids and returns the business_id with its associated rating

        :param user_reviews: DataFrame of all reviews submitted by user
        :param visited_restaurant_ids: Series of all business_id's that a user visited
        :return: DataFrame of business_id's and associated star ratings
        """
        review_ratings = user_reviews[
            user_reviews["business_id"].isin(visited_restaurant_ids)
        ]\
        .sort_values(by="business_id")["stars"]\
        .reset_index(drop=True)

        return review_ratings
    
    def transform_new_restaurants(self):
        """
        transform_new_restaurants takes the scope of the user's desired recommendations
        into account and transforms those restaurants into one hot encoded matrix
        
        :return: feature_matrix of businesses one hot encoded from the main business DataFrame
        """

        number_of_businesses = self.business_data["business_id"].shape[0]
        recommendation_features = np.zeros((number_of_businesses, self.categories.shape[0]))

        for i, j in zip(range(recommendation_features.shape[0]), self.business_data.iterrows()):
            frame_index = i
            raw_categories = self.business_data.iloc[frame_index]["categories"]
            recommendation_features[frame_index] = self.__process_feature_vector(raw_categories)

        return recommendation_features

    def __process_feature_vector(self, business_categories):
        """
        __process_feature_vector takes in set of business_categories, iterates through it and 
        returns a vector with each index corresponding to a word in self.categories

        :param business_categories: set of business categories
        :return: 1-dimensional array of d category features  
        
        """
        business_features = np.zeros(len(self.categories))
        for word in business_categories:
            business_features[self.word_location[word]] = 1
        return business_features