<p align="center">
 <img width="500" src="https://www.swansea.ac.uk/_assets/images/logos/swansea-university-2017.en.png">
</p>

## General Summary
Hello, and welcome.

Within this repository, you will find that there are, in fact, two parts, in short:
1. There is the part concerned with preparing the regional data for use with XAI techniques to see what insights can be divulged regarding the effects of control measures on the Rt.

2. Then there is the part concerned with using just the Rt created during the former's preprocessing. The Rt is then mimicked by an Agent using a limited number of potential actions. Then the Agents results are used to see if a future action can be predicted based on a previous amount of actions (e.g. 7) using **RNN** and **Regressors**.


For 1) you can find the files in **code/preprocessing/** all the files here are used for preparing the data and calculating the Rt using the Cumulative Cases.

For 2) it is all contained within in **code/machine learning/rtPredictor.py**, everything from determining each region's potential actions within the data, running the Agents, and predicting the following action using Machine Learning techniques.


## Requirements
To run the files, you will need the following:

* **Python** (development was performed on 3.7.4)

* **A Linux based system** (or if using Windows, use something like Windows Subsystem for Linux)

* **Jupyter Notebook** (only if you want to look at the XAI parts)

* Modules:
  * **copy** - Used for creating deepcopies of objects
  * **csv** - Used for reading in and saving files
  * **datetime** - Used primarily for filenames to prevent overwriting
  * **imblearn** - Contains a large number of balancing techniques, not essential
  * **matplotlib** - Used for producing the graphs in *images/*
  * **multiprocessing** - Used for running the Rt calculation for each region in parallel
  * **pandas** - Used in *rtPredictor.py* for reading in CSVs, Quantile Bucketing and Dataframes 
  * **scipy** - Used in *compute_Rt.py* for calculating the gamma
  * **shap** - Used in the Jupyter Notebooks in *machine learning/* for its XAI explainer
  * **sklearn** - Used for the metrics, preprocessing and Machine Learning methods
  * **statistics** - Used for its robust mean() function
  * **sys** - Used in the main function for the core files
  * **tensorflow** - Used for its CNN and RNN components
  * **xgboost** - Used for its Machine Learning techniques that work well with SHAP





## Weather Data
The source of all-weather data is https://rp5.ru/

To get updated weather data, then you need to do the following:
1) Search for the region or place you want (e.g. Cardiff, London, etc.)
![image](https://user-images.githubusercontent.com/44265768/113139356-eb57b500-921e-11eb-9b35-0e1ee2a5ff9e.png)
2) Click on the one most pertinent to where you want the data from; you will then be taken to a weather forecast. Click on *"See on map"*.
![image](https://user-images.githubusercontent.com/44265768/113139833-8781bc00-921f-11eb-8b04-f95c22caac11.png)
3) This will take you to a map of the world, but it will now show you the marked locations of nearby weather stations. Click on the one closest to where you want data.
![image](https://user-images.githubusercontent.com/44265768/113140106-df202780-921f-11eb-8a34-20ecb4b002f1.png)
4) Click on *"Weather archive at the weather station"*
![image](https://user-images.githubusercontent.com/44265768/113140313-2b6b6780-9220-11eb-99f4-4dbccf8b44c8.png)
5) Click on the *"Download weather archive"* tab, enter the date range you want and for *Format* select *CSV*, then select *UTF-8* for the encoding.
6) then click *"select to file GZ (archive)"*, after pressing that, a *"Download"* hyperlink should appear, which you then press on.
![image](https://user-images.githubusercontent.com/44265768/113140687-a7fe4600-9220-11eb-9f29-e1e6ef612adf.png)
7) Download the file, extract it so that you have the CSV, rename it so that you know which region it is referring to and place it in the corresponding *data/weather/* folder
8) Then run *code/preprocessing/process_weather.py*, which will clean up the CSV, then determine the humidity and temperature by averaging the readings throughout the day.
![image](https://user-images.githubusercontent.com/44265768/113141385-8baed900-9221-11eb-8469-1b3ee1cd01ff.png)
9) Make sure to check the weather data and the command line. As can be seen, I am getting errors due to missing data, but this is not a concern due to the missing data being few and far between and does not compromise the data for the day. Due to the weather data being taken rather frequently, some have it for every hour or every half-hour. Therefore a small number of missing points for a day is acceptable).

![image](https://user-images.githubusercontent.com/44265768/113141750-024bd680-9222-11eb-897f-0e2459173d56.png)

However, while finding weather stations to use for this project, I did come across many that had significant gaps in the data. This ranged from weeks to, in some cases, months, which renders those useless for our purposes.
For example, while getting weather data for a regional Wales dataset, the following gave incomplete data:

![image](https://user-images.githubusercontent.com/44265768/113142256-961da280-9222-11eb-9cec-48bb32f03cbd.png)

