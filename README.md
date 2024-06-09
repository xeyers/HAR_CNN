This repository contains some convolutional neural network models trained for Arificial Intelligence and Expert Systems subject.

If you are not one of my teachers and are just looking, look at your own risk, this code probably won't explode your computer but keep in mind that I am no specialist at this.

# Important note:
Please keep in mind that the accuracy of those models is questionable at best. While the final one shows around 95 % accuracy the amount of data it was trained on was fairly limited. It was trained on data from WISDM dataset (just a small part of it as to not create imbalances, for more info look into any Jupyter Notebook file) and whatever I was able to record myself (and as a single person I wasn't able to record that much). If this project will ever be expanded to create something meaningful much more data will be needed.

# Contents:
## Models
Folder with models traing throughout the research, each described below

***

### first_model_WISDM_only.keras (original name: my_model_ten_lepszy.keras)
Model trained only on data from university of Fordham. Every axis was used. This model seems to perform well on their data and not so great on the data recorded by me.

***

### walking_stairs_my_data.keras (my_model_na_moim_mini.kera)
Model trained on the small amount of data I was able to record myself.
It only recognizes 3 activities:
- Walking
- Going upstairs
- Going downstairs
On tested data it had not too bad accuracy around 70 % but keep in mind it was trained and tested on a very small amount of data so its accuracy might not be very reliable.

***

### mixed data model with all axes
That model unfortunately got lost in action. Its accuracy was very meh but it sparked a solution used in the final model.

***

### mixed_model_no_x.keras (NO X AXIS) (number of foreign data matched to the smallest amount in my own dataset)(original name: model_mieszany_bez_x.keras)
I came to conclusion that due to other subjects at uni I am unable to record a lot of data so I decided to mix my own with data from university of Fordham dataset. In this model I matched number of entries for lacking activities to the smallest number of entries in my own dataset. This model is after the realization that X axis does more bad than good and could be regarded as unwanted noise.

***

### mixed_model_no_x_more_data.keras (NO X AXIS) (original name: model_mieszany_bez_x_wiecej_danych.keras)
This model is the same as the previous one but this time I used more data from university of Fordham dataset for all activities which resulted in better overall accuracy

### first_model_WISDM_only_no_x.keras (NO X AXIS) (original name: model_pierwotny_ale_bez_x.keras)
As the name suggest this is literally the first model but trained on the data with no X axis. In the research it was tested on test data from university of Fordham dataset but also on the mixed one with my own. On its own it showed same 95 % accuracy. On the mixed one it only mixed up standing with sitting which was to be expected due to nature of both activities. This is the final model.

***

## Data
Folders containing data used to train/test specific models, each described below (WISDM are too big to include on GitHub, text file with link to them was provided instead)
- WISDM_ar_v1.1 - archive with lab made dataset from university of Fordham used in almost all models
- WISDM_at_v2.0_real_life - Bonus. Dataset mentioned in AI_HAR_TESTING.ipynb description below. I don't recommend using this as it yields bad results/
- ultimate.csv - used for all mixed models (all axes)
- Ultimate_no_x.txt - used for all mixed model (no X axis) and for testing final model

***

## Code
- AI_HAR.ipynb - Code for the first model with all axes, university of Fordham dataset only
- AI_HAR_2.ipynb - Bonus. It was used to train model (not included) on university of Fordham dataset from real life (previously mentioned on was lab made). The results were so abysmal that the model was discarded altogether.
- AI_HAR_TESTING.ipynb - Code that was used for all models with no X axis (technically it was used for all models except first one but it was modified to run 2 axes only. If you dear reader wish to run any 3 axes model I suggest modifying "AI_HAR" file). It must be edited appropriately to run

***

## How to use
Each Jupyter Notebook file contains neccessary comments to figure that out, if I did it by watching a YouTube video you can too.
Generally swap path to the data you want (make sure it has right number of axes) and run all cells related to data stuff. Do not run cell related to model training if you only want to test the model (you will most likely have to load model as in all of the code files it uses model trained in cell above, you can easily find how to do that in google, I don't want to provide how to do it because it might change in the future). Then after loading model from provided files all there is left to do is to run the last cell for testing.

***

Useful resources (both are a bit outdated so one thing related to pandas package will not work, I don't remember which one but it is fixed in my code. Second thing to keep in mind both of resources given below don't explore 2 axes only topic nor do they test on data from outside of university of Fordham dataset):
- [KGP Talkie's video about this topic](https://www.youtube.com/watch?v=lUI6VMj43PE)
- [Ravi Raj & Andrzej Kos report covering this topic](https://www.nature.com/articles/s41598-023-49739-1#Tab1)

***

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

***

Everything was done in Python 3.12