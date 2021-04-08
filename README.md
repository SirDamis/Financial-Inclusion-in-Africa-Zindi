# Financial-Inclusion-in-Africa-Zindi 
Financial Inclusion remains one of the main obstacles to economic and human development in Africa. For example, across Kenya, Rwanda, Tanzania, and Uganda only 9.1 million adults (or 13.9% of the adult population) have access to or use a commercial bank account.
Traditionally, access to bank accounts has been regarded as an indicator of financial inclusion. Despite the proliferation of mobile money in Africa, and the growth of innovative fintech solutions, banks still play a pivotal role in facilitating access to financial services. Access to bank accounts enable households to save and facilitate payments while also helping businesses build up their credit-worthiness and improve their access to other finance services. Therefore, access to bank accounts is an essential contributor to long-term economic growth.

## Objective
The objective is to create a machine learning model to predict which individuals are most likely to have or use a bank account. The models and solutions developed can provide an indication of the state of financial inclusion in Kenya, Rwanda, Tanzania and Uganda, as well as provide insights into some of the key demographic factors that might drive individuals’ financial outcomes.

## Data
Data was gotten from [Zindi](https://zindi.africa/competitions/financial-inclusion-in-africa/data). The main dataset contains demographic information and what financial services are used by approximately 33,610 individuals across East Africa. This data was extracted from various Finscope surveys ranging from 2016 to 2018.

## Packages
- Numpy
- Pandas
- Seaborn
- Streamlit
- Catboost
- XGBoost
- LightGBM


## Evaluation: The prediction achieved accuracy of 88%.

## Models: An ensemble of three models(XGBoost, Catboost, LightGBM) was used. The model was trained using StratifiedKFold on split of 25. Training was done on CPU. Saved model as a pickle file.

## Deployment
The model was deployed using Streamlit. Deployed web app can be accessed using this [link]()
