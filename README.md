![Field Crops Classification](/_assets/banner_logo.png)

# AgroLuege cx5

This project is a part of the [CX5 AgroLuege Group](https://gitlab.fhnw.ch/thomas.mandelz/AgroLuege) at [Data Science FHNW](https://www.fhnw.ch/en/degree-programmes/engineering/bsc-data-science).

Our Challenge-X project is inspired by and adapted from the [ZueriCrop Paper](https://arxiv.org/abs/2102.08820) (Turkoglu et al., 2021).

## Project Status: Completed

## Project Objective

### Current Challenge

In the current scenario, farmers are required to manually input their land areas on AgriPortal to receive state direct payments. This process not only consumes time but also poses the risk of errors.

### Our Vision

Our goal is to provide farmers with more time for their actual work. We leverage the power of Deep Learning and satellite imagery to automatically identify which crops are cultivated on specific land areas. This cutting-edge technology is designed not only to save time but also to minimize error susceptibility.

We have developed advanced Deep Learning models capable of classifying various crops on agricultural land. By analyzing satellite images, we can efficiently and accurately gather information about the cultivation on these areas.

### Methods Used

* Deep Learning
* Computer Vision
* Explorative Dataanalysis
* Data Visualization
* Remote Sensing
* Crop Classification
* Big Data

### Technologies

* Python
* PyTorch
* torchgeo
* wandb
* numpy
* pandas
* seaborn
* eodal
* sentinelhub
* geopandas
* h5py
* rasterio
* MSConvStar Model



## Result Reports

Here you find our reports of the various experiments:
* [Reproduction Report of of ZueriCrop Paper](https://wandb.ai/agroluege/MS-Convstar/reports/Reproduktion-Z-riCrop-ETHZ--Vmlldzo2MjcyNTA1)
* [Reduction of Noise](https://wandb.ai/agroluege/MS-Convstar/reports/Reduktion-der-Sentinel-Bilder--Vmlldzo2NDE2MjQ3)
* [Increasing Model Complexity](https://wandb.ai/agroluege/MS-Convstar/reports/MSConvStar-erg-nzen-mit-9-Layers--Vmlldzo2NDE2MTcy)
* [Initialisation and Tuning experiments](https://wandb.ai/agroluege/MS-Convstar/reports/Normalisierung-und-Lernrate--Vmlldzo2NDE2Mjcz)

## Pipenv for Virtual Environment

### First install of Environment

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv install`
* Restart VS Code
* Choose the newly created "AgroLuege" Virtual Environment python Interpreter

### Environment already installed (Update dependecies)

* open `cmd`
* `cd /your/local/github/repofolder/`
* `pipenv sync`

## Overview Folder

| Folder            | Subfolders                        | Description                             |
|-------------------|----------------------------------|-----------------------------------------|
| data              |                                  | Directory for data processing, handling and download              |
| eda               |                                  | Exploratory Data Analysis files        |
| models            |                                  | Directory for model architecture files   |
| raw_data          | BernCrop, LANDKULT, ZueriCrop    | Raw data files like HDF5, labels and vector features                        |
|   └── BernCrop    | tiles                            | folder for single tiles of BernCrop (only local)          |
|   └── LANDKULT    | data, metadata, symbol, table    | Raw data for LANDKULT data, ground truth and vector features          |
|       └── data    | tiles                            | Data shapes for tiles         |
|       └── metadata| layer                            | Metadata for LANDKULT          |
|   └── ZueriCrop   |                                  |Raw data for ZueriCrop (only local)     |
| scripts           |                                  | Scripts and code files                 |
| src               |                                  | Source code files for DL pipeline            |



## Contributing Members

* **[Daniela Herzig](https://gitlab.fhnw.ch/daniela.herzig)**
* **[Thomas Mandelz](https://github.com/tmandelz)**
* **[Jan Zwicky](https://github.com/swiggy123)**
