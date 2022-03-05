"""
Author: Brandon Golshirazian
Date: 12/13/2021
Description: File to preprocess json data into parquet files. Apache Parquest files
are a columnar based file format that works well with quick compression as well.
Since the data we're dealing with is large (11gb total), using this file format reduces 
time significantly (for e.g review dataset 20-30 minutes becomes 40 seconds)

Please be patient with script, should take about 20-30 minutes in total :)
"""
import pandas as pd
from halo import Halo


# business data
text = "Processing yelp_academic_dataset_business.json to parquet format ⌚"
spinner = Halo(text=text, spinner='dots2')
spinner.start()
yelp_business_data = pd.read_json("../data/raw/yelp_academic_dataset_business.json", lines=True)
yelp_business_data.to_parquet("../data/pre-processed/yelp_business_data.parquet", engine='pyarrow')

spinner.stop_and_persist(symbol="✅", text="yelp_business_data.parquet exported!")

# user data
text = "Processing yelp_academic_dataset_user.json to parquet format ⌚"
spinner = Halo(text=text, spinner='dots2')
spinner.start()
yelp_user_data = pd.read_json("../data/raw/yelp_academic_dataset_user.json", lines=True)
yelp_user_data.to_parquet("../data/pre-processed/yelp_user_data.parquet", engine='pyarrow')

spinner.stop_and_persist(symbol="✅", text="yelp_user_data.parquet exported!")

# free up some memory because next dataset may crash if it doesn't have enough ram
to_delete = [yelp_user_data, yelp_business_data]
del yelp_business_data
del yelp_user_data
del to_delete

# reviews data set (split into 2 files)
print("Processing yelp_academic_dataset_review.json to parquet format ⌚, please be patient 🐢!")
text = "\tStep (1/2): Splitting 🪓"
spinner = Halo(text=text, spinner='dots2')
spinner.start()
yelp_reviews_data = pd.read_json("../data/raw/yelp_academic_dataset_review.json", lines=True)
reviews_first_half = yelp_reviews_data[:len(yelp_reviews_data)//2]
reviews_second_half = yelp_reviews_data[len(yelp_reviews_data)//2:]

spinner.stop_and_persist(symbol="\t✅", text="Data loaded and split!")

text = "\tStep (1/2): Exporting 💾"
spinner = Halo(text=text, spinner='dots2')
spinner.start()
reviews_first_half.to_parquet("../data/pre-processed/yelp_academic_dataset_reviews_part1.parquet", engine='pyarrow')
reviews_second_half.to_parquet("../data/pre-processed/yelp_academic_dataset_reviews_part2.parquet", engine='pyarrow')

spinner.stop_and_persist(symbol="\t✅", text="yelp_academic_dataset_reviews_part1.parquet & yelp_academic_dataset_reviews_part2.parquet exported!")

print("🎉 Your data is ready to be used! 🎊")