# SALE OFFERS PREDICTION IN THE SPANISH DAY-AHEAD ELECTRICITY MARKET


**TFM - DATA SCIENCE MASTER, 26th edition (MAD) - KSCHOOL - July, 2021**

**LUIS MUÑOZ DEL POZO**





## 1. INTRODUCTION

This TFM explores the way **combined cycle** and **hydraulic** units present their bids to the Nominated Electricity Market Operator (**OMIE**) every day in the Spanish day-ahead electricity market, and tries to find out a method to predict these offers with **ML techniques**, and only **public information**.



## 2. OBJECTIVE

The first objective of this TMF is to retrieve, understand, and process the public information of the sale offers that OMIE publishes every day after a confidentiality period of 90 days.

The second objective of this work is to find out a way to predict the sale offers for combined cycle, and hydraulic units.



## 3. INSTRUCTIONS TO GET DATA AND RUN THE CODE

This TFM has been developed with **PYTHON version 3.8.5** and **UNIX** evironment.

The TFM is divided in the following parts:


**1. Repo files in https://github.com/luisdelpozo/TFM**.
    The files stored in this repo are the following:

- **00_MEMORIA.pdf**. This document explains deeply the TFM structure, main objectives, description of the raw data, methodology, summary of main results, conclusions, and the frontend user manual.

- **Notebooks and functions:**
   - **Notebooks from 01 to 04**. Notebooks where raw data are retrieved, studied, and managed in a first phase, previous to the final data preparation in the model notebooks. Note that, in order to run the model notebooks, it is not necessary to run these notebooks, since the output data of them have been storaged locally (in **"./Data_Input/"** folder, as explained above). This notebooks use some UNIX terminal commands, so if they are runned in Windows, slights modifications would be needed.
   
   - **Notebooks from 10 to 18**. Notebooks where combined cycle unit bids are studied, model data are prepared, and a prediction models are created and tested against a naive model.
   
   - **Notebooks from 20 to 23**. Notebooks where hydraulic unit bids are studied, data are prepared, and a prediction model is created and tested against a naive model.
   
   - **Notebook 30_UNITS_Bid_Comparison.ipynb**. Notebook where different unit bids are compared.
   
   - **Notebook 40_FRONTED.ipynb**. Notebook where the frontend is developed:
      - **Python files from 41 to 45**. Files to be able to execute the frontend.
      
   - **TFM_PredCurve_Tools.py**. In this file, different functions are stored to be used in the notebooks.
   
   - **TFM_Requirements.txt**. In this file, all libraries and requirements to run the notebooks are listed. To install all the required libraries and dependencies it is necessary to run the following code:
   
   > pip install TFM_Requirements.txt


**2. Raw data, input data of the notebooks, output data from the notebooks, and pickle files**. All these files have been stored in Google Drive in the following link: https://drive.google.com/drive/folders/1HPIvBoqFMj2icafklc6Eh3OAt1r2OqV6?usp=sharing

In order to be able to run the code in the notebooks, the folder, subfolders and files stored in Google Drive must be downloaded in the same folder where REPO files are, and with the same structure as they are in the link. After downloading them, the “.zip” files (**Pickle_Models.zip** and **Raw_Data.zip**) must be unzipped.

At the end of this process, the folders/file structure of the TFM must be the following:

![image1](https://github.com/luisdelpozo/TFM/blob/main/TFM_FILE_STRUCTURE.jpg "TFM folders/files structure.")


