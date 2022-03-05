# Content Based Recommendation System Using the Yelp Dataset

## Introduction

Using the data set provided openly by yelp we've built a recommendation system that uses content based information filtering.

The Yelp dataset is pretty large and mostly contains data from metropolitan cities within a few states in the U.S & Canada.

To run the project properly your folder/file structure is going to end up looking something like this...

```
├── README.md
├── data
│   ├── pre-generated
│   │   └── association_matrix.csv
│   ├── pre-processed
│   │   ├── yelp_academic_dataset_reviews_part1.parquet
│   │   ├── yelp_academic_dataset_reviews_part2.parquet
│   │   ├── yelp_business_data.parquet
│   │   └── yelp_user_data.parquet
│   └── raw
│       ├── yelp_academic_dataset_business.json
│       ├── yelp_academic_dataset_review.json
│       └── yelp_academic_dataset_user.json
├── yelp-exploration.ipynb
├── model-testing.ipynb
├── scripts
│   └── requirements-dump.sh
├── src
│   ├── content_based_recommender.py
│   ├── data_preprocessing.py
│   └── feature_processing.py
├── wsl_requirements.yml
└── mac_requirements.yml
```

so follow the steps below and you'll be up and running in no time!

---

## Overview

Here's a quick overview of the files that you'll want to look into/interact with in this project...

**_Model Building_**

- **yelp-exploration.ipynb:** contains visualizations and some insight into yelp's user, business, and review datasets
- **model-testing.ipynb:** has a working sample of the recommender system mode (ContentBasedRecommender) built with a sample target user_id
- **content_based_recommender.py:** Contains actual implementation of the model building to provide recommendations for a specified user
- **feature_processing.py:** contains class and other useful methods that came in handy for pre-processing data for model building/training

**_Data Processing/Other Scripts_**

- **data_preprocessing.py:** If you choose to preprocess the data yourself after downloading from yelp's website, then you run this script to convert those clunky json files into fast and sleek parquet files
- **requirements-dump.sh:** If you make any changes to the dependencies using `conda install your_package_here` then run this script to dump the new requirements (not necessary to run)

---

## Getting Started

### Step 1. Installing Dependencies

**Make sure you have Ancaconda installed**

If you already have that installed, go ahead and run this script to create a conda environment to run the code with all of its dependencies

_on WSL (Windows) / Ubuntu:_

`conda env create -f wsl_requirements.yml`

`conda activate yelp-recommend` (the environment) and you should be good to go.

Just to double check your environment you can run the following code in a jupyter notebook to see which environment your python is coming from

```
import sys
print(sys.path)
```

_on macOS_

`conda env create -f mac_requirements.yml`

If you're on macOS you're not quite done yet. To access the environment within the jupyter notebook run this script to make your environment selectable as a kernel in jupyter

`python -m ipykernel install --user --name yelp-recommend --display-name "Python (yelp-recommend)"`

**When you're in jupyter make sure to go to "Kernel" > "Change Kernel" > "Python (yelp-recommend)"**

---

### Step 2. Getting/Pre-processing the Data

You have a couple options here:

**Option 1.** Download Yelp's data online and pre-process it yourself:

1. [Download yelp files from here](https://www.kaggle.com/yelp-dataset/yelp-dataset)
2. Drop the files in the `data/raw/` folder
3. Navigate to the `src/` folder and run `python3 data_preprocessing.py` (you must be in the src folder to run this script) **Warning: This will take anywhere from 20-40 minutes depending on your computer and may even crash if you don't have a lot of ram**

**Option 2.** Download the processed files so you don't have to preprocess them yourself

1. [Download the files from this drive](https://drive.google.com/drive/folders/1l2-3dKBXINEpUFgQG8RGpaU9RMWpSBNp?usp=sharing)
2. Just drop them in the `data/pre-processed` folder

Lastly, download the association matrix csv file [from here](https://drive.google.com/drive/folders/1l2-3dKBXINEpUFgQG8RGpaU9RMWpSBNp?usp=sharing) as well and drop it in the `data/pre-generated` folder

Now the data is all ready to go! 🎉

---

### Step 3. Running the model and any other explorative code

**Exploration**

To see visualizations and an overview/explanation of the data start `jupyter notebook` and open up the yelp-exploration.ipynb notebook.

This contains visualizations on the data. It also takes a deeper look at the features (business categories) we use to train the model on.

**Model Testing/Running**

To run the model and test it out start jupyter once again (like above), and open up the model-testing.ipynb notebook

This notebook contains a sample usage of the model developed to build the content based recommender system. Run the code to get an idea of how it works, but look into the package files _content_based_recommender.py_ & _feature_processing.py_ to get a better idea of how the model works.
