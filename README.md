This repository contains some convolutional neural network models trained for Arificial Intelligence and Expert Systems subject.

If you are not one of my teachers and are just looking, look at your own risk, this code probably won't explode your computer but keep in mind that I am no specialist at this.

# Important note:
Please keep in mind that the accuracy of those models is questionable at best. While the final one shows around 95 % accuracy the amount of data it was trained on was fairly limited. It was trained on data from WISDM dataset (just a small part of it as to not create imbalances, for more info look into any Jupyter Notebook file) and whatever I was able to record myself (and as a single person I wasn't able to record that much). If this project will ever be expanded to create something meaningful much more data will be needed.

# Contents:
## Models
Folder with models traing throughout the research, each described below

### first_model_WISDM_only.keras (original name: my_model_ten_lepszy.keras)
Model trained only on data from university of Fordham. Every axis was used. This model seems to perform well on their data and not so great on the data recorded by me.

### walking_stairs_my_data.keras (my_model_na_moim_mini.kera)
Model trained on the small amount of data I was able to record myself.
It only recognizes 3 activities:
- Walking
- Going upstairs
- Going downstairs
On tested data it had not too bad accuracy around 70 % but keep in mind it was trained and tested on a very small amount of data so its accuracy might not be very reliable.

### mixed data model with all axes
That model unfortunately got lost in action. Its accuracy was very meh but it sparked a solution used in the final model.

### mixed_model_no_x.keras (NO X AXIS) (number of foreign data matched to the smallest amount in my own dataset)(original name: model_mieszany_bez_x.keras)
I came to conclusion that due to other subjects at uni I am unable to record a lot of data so I decided to mix my own with data from university of Fordham dataset. In this model I matched number of entries for lacking activities to the smallest number of entries in my own dataset. This model is after the realization that X axis does more bad than good and could be regarded as unwanted noise.

### mixed_model_no_x_more_data.keras (NO X AXIS) (original name: model_mieszany_bez_x_wiecej_danych.keras)
This model is the same as the previous one but this time I used more data from university of Fordham dataset for all activities which resulted in better overall accuracy

### first_model_WISDM_only_no_x.keras (NO X AXIS) (original name: model_pierwotny_ale_bez_x.keras)
As the name suggest this is literally the first model but trained on the data with no X axis. In the research it was tested on test data from university of Fordham dataset but also on the mixed one with my own. On its own it showed same 95 % accuracy. On the mixed one it only mixed up standing with sitting which was to be expected due to nature of both activities. This is the final model.

## Data
Folders containing data used to train/test specific models, each described below
(to be added)

## How to use
Each Jupyter Notebook file contains neccessary comments to figure that out, if I did it by watching a YouTube video you can too (link will be here)

## Required packages
- tensorflow
- keras
- pandas
- numpy
- matplotlib
- sklearn
- scipy
- mlxtend
- seaborn

Everything was done in Python 3.12