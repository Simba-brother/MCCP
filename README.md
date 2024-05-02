## Website for data set collection

| Domain   | $Task_A$                                                                                       | $Task_B$                                                                                                         |
|----------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Car      | https://www.kaggle.com/datasets/darshan1504/car-body-style-dataset                          | https://pan.baidu.com/s/1xeYXXIp0V-llV1c9IEqk-w (password:zq4s)                                               |
| Flower   | https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc | https://www.kaggle.com/datasets/alxmamaev/flowers-recognition                                                 |
| Food     | https://www.kaggle.com/datasets/imbikramsaha/food11                                         | https://www.kaggle.com/datasets/manishkc06/food-classification-burger-pizza-coke?select=Training_set_food.csv |
| Fruit    | https://www.kaggle.com/datasets/nguyenductai243/10-fruit                                    | https://www.kaggle.com/datasets/alibaloch/vegetables-fruits-fresh-and-stale                                   |
| Sport    | https://www.kaggle.com/datasets/gpiosenka/sports-classification                             | https://www.kaggle.com/datasets/rishikeshkonapure/sports-image-dataset                                        |
| Weather  | https://www.kaggle.com/datasets/jagadeesh23/weather-classification                          | https://www.kaggle.com/datasets/jehanbhathena/weather-dataset                                                 |
| Animal_1 | https://www.kaggle.com/datasets/biancaferreira/african-wildlife                             | https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set/code                          |
| Animal_2 | https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda      | https://www.kaggle.com/datasets/shiv28/animal-5-mammal                                                        |
| Animal_3 | https://www.kaggle.com/datasets/enisahovi/cats-projekat-4                                   | https://www.kaggle.com/datasets/anshulmehtakaggl/wildlife-animals-images?select=cheetah-resize-224            |

## Description of the core document
./model_combination.py<br>
Codes for MCCP joins Teacher Models

./MCCP_retrain.py<br>
Codes for training and evaluation of the MCCP methodology

./HMR_retrain.py<br>
Codes for training and evaluation of the HMR methodology

./CFL_retrain.py<br>
Codes for training and evaluation of the CFL methodology

./Dummy.py<br>
Codes for evaluation of the Dummy methodology

./DecisionTree.py<br>
Codes for evaluation of the DecisionTree methodology

./LogisticRegression.py<br>
Codes for evaluation of the LogisticRegression methodology

./utils.py
1. Tool Code File<br>
Used in e.g. deleteIgnoreFile,makedir_help,saveData
2. Segmentation and organisation of native datasets

./statistical.py<br>
For statistical work on data, e.g. calculation of Win/Tie/Lose and correlation

./model_prepare folder<br>
It is used to train the teacher model with the aim of obtaining a well-performing teacher model.

./cut_model.py<br>
The code is used to segment the teacher model and obtain the cut teacher model in order to obtain the feature layer output of the teacher model.

./draw.py<br>
Code for statistical data plotting.

./exp_data<br>
The code is used to store variables during the experiment, such as sampled data

./exp_image<br>
Used to store plotted statistical graphs

./DatasetConfig.py<br>
Main configuration file

./DatasetConfig_2.py<br>
Another configuration file

./requirements.txt<br>
Project runtime dependency packages


## Download addresses of other Datasets and Models
Since the data set and model are large and the platform has limitations, we upload the remaining data sets and models to the link below. If necessary, you can download them through the link below.
https://pan.quark.cn/s/09b446b8f944