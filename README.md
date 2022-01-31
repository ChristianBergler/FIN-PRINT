# FIN-PRINT a fully-automated multi-stage deep-learning-based framework for the individual recognition of killer whales


## General Description
FIN-PRINT is a fully automated, multi-stage, deep learning framework for killer whale individual classification, composed of multiple sequentially ordered machine (deep) learning sub-components and with the aim to automatize and support the analysis of killer whale photo-identification data. At first, object detection is performed to identify unique killer whale markings (dorsal fin and saddle patch). In a second step, all previously detected natural killer whale markings are extracted to equally sized and square images.  The third and next step involves a data enhancement procedure by filtering between images including valid and invalid natural markings of previous outputs. The fourth and final step performs multi-class individual classification in order to distinguish between the 100 most commonly photo-identified killer whales.

FIN-DETECT, the deep-learning-based killer whale detection network, is used first to extract the respective image regions of interest. FIN-EXTRACT, the image extraction software, extracts 512 x 512 large sub-images, based on the previous detections. VVI-DETECT, a binary-class Convoluational Neural Network integrates the mentioned data enhancement procedure by filtering inappropriate sub-images. FIN-IDENTIFY, also a Convolutional Neural Network, addressing mulit-class classification for killer whale individual recognition. All modules, the requried software libraries, file- and data-structures, network training configurations, and final model evaluation scenarios are described in detail within the subsequent guidelines.  

## Reference
If FIN-PRINT is used for your own research please cite the following publication: FIN-PRINT a fully-automated multi-stage deep-learning-based framework for the individual recognition of killer whales 

(https://www.nature.com/articles/s41598-021-02506-6)

```
@article{Bergler-FP-2021,
author = {Bergler, Christian and Gebhard, Alexander and Towers, Jared and Butyrev, Leonid and Sutton, Gary and Shaw, Tasli and Maier, Andreas and Noeth, Elmar},
year = {2021},
month = {12},
pages = {23480},
title = {FIN-PRINT a fully-automated multi-stage deep-learning-based framework for the individual recognition of killer whales},
volume = {11},
journal = {Scientific Reports},
doi = {10.1038/s41598-021-02506-6}
}
```
## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

# FIN-DETECT

## Python, Python Libraries, and Version
FIN-DETECT is a deep learning algorithm which has been implemented in Python (Version=3.8) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.8.1, TorchVision=0.9.1). Moreover it requires the following Python libraries: Pillow, Terminaltables, TensorboardX, Matplotlib, Tqdm, next to a portion of Python libraries already integrated within the standard Python repository (recent versions).

## Required Filename Structure for Training
In order to properly load and preprocess your own animal-specific data to train the network you need to prepare the filenames of your image data, next to the bounding box files for each image, in order to fit the following template/format:

Filename Image-Template: "SPECIES"\_"YYYY-MM-DD"\_"PHOTOGRAPHER"\_"LOCATION"\_"ID".(jpg, JPG, png) 

Filename Bounding-Box-Template: "SPECIES"\_"YYYY-MM-DD"\_"PHOTOGRAPHER"\_"LOCATION"\_"ID".txt 

**1st-Element**: SPECIES = Information about the corresponding animal species

**2nd-Element**: YYYY-MM-DD = Date when the image was taken, following the format Year, Month, and Day

**3rd-Element**: PHOTOGRAPHER = Name of the Photographer

**4th-Element**: LOCATION = Name of the Location where the Image was taken

**5th-Element**: ID = Unique ID (natural number) to identify the respective pair between image and bounding-box file

Due to the fact that the underscore (_) symbol was chosen as a delimiter between the single filename elements please do not use this symbol within your filename except for separation.

<ins>**Example of valid filenames**:</ins>

*KillerWhale_2022-01-28_ChristianBergler_VancouverIsland_1234.JPG*

**1st-Element - SPECIES**: = KillerWhale

**2nd-Element - YYYY-MM-DD**: = 2022-01-28

**3rd-Element - PHOTOGRAPHER**: ChristianBergler

**4th-Element - LOCATION**:  = VancouverIsland

**5th-Element - ID**:  = 1234

## Required Directory Structure for Training
FIN-DETECT is capable of performing its own training, validation, and test split if there is not an existing one. Based on a given image folder (--image_folder option, needs to contain all image files, together with the bounding box ground truth annotation files) a new data partitioning will be conducted. Otherwise an existing data split has to exist. Within the config/info.data file the respective data split has to be defined. In case of the machine-based data partitioning everything will be done automatically. In case of an existing data split the train.txt, valid.txt, and test.txt path has to be adjusted/updated within the config/info.data file.

<ins>**Example of train.txt, valid.txt, and test.txt structure:**</ins>

```
*ImagePath/ImageFile1.JPG\
ImagePath/ImageFile2.JPG\
ImagePath/ImageFile3.JPG\
ImagePath/ImageFile4.JPG\
(...)*
```

All bounding box files have to have the same filename as the original image, just a different filename ending (.txt). Moreover, the machine generated data split (only active if the --image_folder is set) offers an option that the validation and test partition does not contain images from the same phototgrapher and date. Bounding box ground truth files can be generated using the YOLO-MARK toolkit which is publicly available here: 

https://github.com/AlexeyAB/Yolo_mark

Each bounding box file (.txt) contains one line per annotated bounding box within a specific image containing the following information:

*ClassIndex x-coor y-coor bb-width bb-height*

ClassIndex is defined within the config/class.names file. The class names are listed per line, while getting an increasing index value assigned, e.g. first line = index 0 = fin, second line = index 1 = no-fin. The x-coordinate (*x-coor*) and y-coordinate (*y-coor*) of the bounding box are the centered coordinates describing the image-specific normalized center of the corresponding bounding box. The image-specific normalized width (*bb-width*) and height (*bb-height*) of the bounding box is describing the width and height of the respective boudning box.


## Network Training
For a detailed description about each possible training option we refer to the usage/code in *train.py* (usage: *train.py -h*), together with the paper descriptions. 

This is just an example command in order to start network training with an existing data split and without an existing data split:

```train.py --debug --log_dir log_dir --summary_dir summary_dir --model_cfg path_to_folder/yolov3-custom.cfg --data_config path_to_folder/info.data --pretrained_weights path_to_folder/yolov3.weights --evaluation_interval 1 --learning_rate 0.0001 --conf_thres 0.5```

```train.py --debug --log_dir log_dir --summary_dir summary_dir --model_cfg path_to_folder/yolov3-custom.cfg --data_config path_to_folder/info.data --pretrained_weights path_to_folder/yolov3.weights --evaluation_interval 1 --learning_rate 0.0001 --conf_thres 0.5 --image_folder image_folder```

The pre-trained weights for YOLOv3, together with the DarkNet-53 architecture as backbone, are publicly available here:

https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master/weights

In this study the experiments were processed using either the *yolov3.weights* or the *darknet53.conv.74* weights, whereas
the *yolov3.weights* led to the best model results.

After a successful training procedure the final model will be saved and stored as .pk (pickle) file within the summary folder.

## Network Testing and Evaluation
During training FIN-DETECT will be verified on an independent validation set. In addition FIN-DETECT will be automatically evaluated on the test set at the very end of the training procedure. In both cases multiple machine learning metrics (x-coor, y-coor, width, height (MSE) and classification (BCE) loss, class accuracy, object confidence, recall, precision, etc.) will be calculated and documented. Besides the traditional logging, all results and the entire training process is also documented and can be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir summary_directory/```

There exist also the possibility to evaluate the FIN-DETECT model on an entire unseen portion of images. The prediction script (*predict.py*) implements a data loading procedure of a given image set, and applies the trained model (either loading the model from a given checkpoint.pth or the final .pk model file) in order to detect the bounding boxes for each image. All bounding box information will be stored within file-specific .txt bounding box files. Moreover, downsized (re-scaled versions of the original image) images, together with the detected and marked bounding boxes, will be additionally stored for reasons of visual inspection. 

<ins>**Example Command:**</ins>

```predict.py --debug --image_folder image_folder --model_cfg path_to_folder/yolov3-custom.cfg --model_path path_to_folder/detector.pk --class_path path_to_folder/class.names --conf_thres 0.8 --nms_thres 0.5 --batch_size 1 --n_cpu 1 --img_size 416 --output_dir output_dir --log_dir log_dir```

# FIN-EXTRACT


## Python, Python Libraries, and Version
FIN-EXTRACT is an algorithm which has been implemented in Python (Version=3.8) (Operating System: Linux) together with the following Python libraries: Pillow, GPSPhoto, ExifRead (recent versions).


## Required Filename Structure
In order to properly load, preprocess, and extract all previously detected (FIN-DETECT) bounding boxes as new, equally resized/rescaled sub-images for further analysis, it is required to provide information about the image set of interest, as well as the corresponding bounding box files. FIN-EXTRACT requires the same filename structure as FIN-DETECT (see above), e.g. 

*KillerWhale_2022-01-28_ChristianBergler_VancouverIsland_1234.JPG* 

and its bounding box file, named:
 
*KillerWhale_2022-01-28_ChristianBergler_VancouverIsland_1234.txt*. 
 
 Based on the amount of bounding boxes, listed within the .txt. file, the corresponding number of sub-images will be extracted, next to an additional .txt file storing the bounding box position-specific values, together with an index indicating a specific bounding box in case of multiple boxes per image. 

## Required Directory Structure for Training
FIN-EXTRACT requires the information about the original images (careful: not the images produced by the *predict.py* file from FIN-DETECT) and the
the data location of the respective bounding box files. Each image filename has to be the same as for the bounding
box filename, except of the filename ending. The script provides the opportunity to store the images in a stand-alone folder,
compared to the bounding box data in another folder. Moreover, there is also the possibility to locate image and bounding box data
within the same folder.

## Image Extraction
For a detailed description about each possible training option we refer to the usage/code in *extract.py* (usage: *extract.py -h*).

Example Command:

```extract.py --debug --log_dir log_dir --original_image_path image_folder --bounding_box_path bounding_box_dir --output_dir output_dir --data_config path_to_folder/info.data```

FIN-EXTRACT generates two output files - sub-image and bounding box file per detected bounding box - following the respective template:

<ins>**Filename Sub-Image-Template**</ins>: 

"SPECIES"\_"YYYY-MM-DD"\_"PHOTOGRAPHER"\_"LOCATION"\_"ID"\_"CLASS"\_cropped\_"EXTRACTION-INDEX"\_"YYYY-MM-DD"\_"HH-MM-SS".(jpg, JPG, png)

e.g. *KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_0_2017-02-21_12-15-55.JPG*

<ins>**Filename Sub-Image Bounding Box Template:**</ins>

"SPECIES"\_"YYYY-MM-DD"\_"PHOTOGRAPHER"\_"LOCATION"\_"ID"\_"CLASS"\_cropped\_"EXTRACTION-INDEX"\_"YYYY-MM-DD"\_"HH-MM-SS".txt

e.g. *KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_0_2017-02-21_12-15-55.txt*

In case neither date or time is available for a specific image these templates will be set to *None*.
In such case, the constraint of not having images from the same photographer and date
within the validation or test set is not being applied for that specific image by all subsequent deep learning training approaches because the required information is just not present. Nevertheless, a data split will be computed, but as said, without considering this specific constraint for those type of images.

# VVI-DETECT

## Python, Python Libraries, and Version
VVI-DETECT is a deep learning algorithm which has been implemented in Python (Version=3.8) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.8.1, TorchVision=0.9.1). Moreover it requires the following Python libraries: Pillow, Pandas, TensorboardX, Matplotlib, next to a portion of Python libraries already integrated within the standard Python repository (recent versions).

## Required Filename Structure for Training
In order to properly load and preprocess your own animal-specific data to train a network in order to distinguish between *Valid Versus Invalid (VVI)* previously detected and equally resized/rescaled target animal-specific sub-images of interest, the filename structure has to follow the original image-specific output structure of FIN-EXTRACT (see above). An image, e.g. *KWT_2017-02-21_MMalleson_ConstanceBank_065.JPG*, together with the corresponding predicted bounding box result (*KWT_2017-02-21_MMalleson_ConstanceBank_065.txt*, e.g. 3 predicted bounding boxes), provided by FIN-DETECT, lead to the following sub-images, created by FIN-EXTRACT:

<ins>**Sub-Images:**</ins>

```
*KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_0_2017-02-21_12-15-55.JPG*\
*KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_1_2017-02-21_12-15-55.JPG*\
*KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_2_2017-02-21_12-15-55.JPG*
```

<ins>**Sub-Image Bounding Boxes:**</ins>

```
*KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_0_2017-02-21_12-15-55.txt*\
*KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_1_2017-02-21_12-15-55.txt*\
*KWT_2017-02-21_MMalleson_ConstanceBank_065_Fin_cropped_2_2017-02-21_12-15-55.txt*
```

In order to train VVI-DETECT the bounding box (.txt) files are not needed. However, the filename structure of the sub-images has to follow the above shown template (same as the output of FIN-EXTRACT), to ensure a correct training procedure.

"SPECIES"\_"YYYY-MM-DD"\_"PHOTOGRAPHER"\_"LOCATION"\_"ID"\_"CLASS"\_cropped\_"EXTRACTION-INDEX"\_"YYYY-MM-DD"\_"HH-MM-SS".(jpg, JPG, png)

## Required Directory Structure for Training
VVI-DETECT is capable of performing its own training, validation, and test split if there is not an existing one. The system either checks whether train.csv, val.csv, and test.csv files already present, or the option *--create_datasets* is set. Otherwise an existing data split has to exist. Using the option *--data_split_dir* specifies the location of the data partitions. In order to generate a proper data split the following folder structure has to be ensured:

Data_Folder (e.g. containing all the sub-folders with the images)
 
    ├── Garbage (name of the directory represents the "Ground Truth Label" for all images within that folder)
    │   ├── KWT_2017-02-26_MMalleson_JuandeFuca_038_Fin_cropped_0_2017-02-26_14-15-04.JPG
    │   ├── KWT_2017-02-26_MMalleson_JuandeFuca_063_Fin_cropped_0_2017-02-26_15-56-34.JPG
    │   └── KWT_2017-03-12_MMalleson_RacePassage_036_Fin_cropped_1_2017-03-12_13-37-44.JPG
    │   └── (...)
    ├── Others (name of the directory represents the "Ground Truth Label" for all images within that folder)
        ├──  KWT_2014-08-04_DMatkin_BartlettRiverSEAK_016_Fin_cropped_3_2014-08-04_23-20-06.JPG
 	    ├──  KWT_2014-08-21_DMatkin_GustavusFlatsSEAK_007_Fin_cropped_0_2014-08-21_13-43-45.JPG
 	    ├──  KWT_2014-09-06_JTowers_ChathamPoint_5918_Fin_cropped_0_2014-09-06_13-22-57.JPG
 	    ├──  KWT_2014-10-26_JTowers_BlackfishSound_8536_Fin_cropped_2_2014-10-26_15-26-20.JPG
 	    ├──  KWT_2014-10-26_JTowers_BlackfishSound_8678_Fin_cropped_1_2014-10-26_17-23-12.JPG
 	    ├──  (...)

<ins>**Example of train.csv, valid.csv, and test.csv structure:**</ins>

```
Label,Filepath\
Garbage,path_to_image_folder/Garbage/KWT_2015-04-02_MMalleson_SouthConstanceBank_1239_Fin_cropped_0_2015-04-02_18-52-16.JPG\
Garbage,path_to_image_folder/Garbage/KWT_2016-04-25_JTowers_CormorantChannel_1108_Fin_cropped_0_2016-04-25_08-50-05.JPG\
Other,path_to_image_folder/Other/KWT_2011-07-29_MGreenfelder_TracyArmSEAK_083_Fin_cropped_0_2011-07-29_17-00-53.JPG\
Garbage,path_to_image_folder/Garbage/KWT_2015-07-13_RAbernethy_RubyRocks_D-6626_Fin_cropped_0_2015-07-13_11-03-55.JPG\
Other,path_to_image_folder/Other/KWT_2013-01-12_GEllis_NeckPoint_0019_Fin_cropped_0_2013-01-12_13-11-44.JPG
(...)
```

## Network Training
For a detailed description about each possible training option we refer to the usage/code in *main.py* (usage: *main.py -h*), together with the paper descriptions. 

This is just an example command in order to start network training with an existing data split and without an existing data split:

```main.py --debug --start_from_scratch --max_train_epochs 500 --augmentation True --max_aug 5 --grayscale False --resnet 34 --lr 10e-3 --max_pool 1 --conv_kernel_size 9 --batch_size 8 --epochs_per_eval 2 --num_workers 6 --early_stopping_patience_epochs 24 --split 0.7 --across_split True --interval 5 --num_classes 2 --img_size 512 --data_dir image_data_directory --data_split_dir data_split_directory --model_dir model_storage_directory --log_dir log_dir --checkpoint_dir checkpoint_dir --summary_dir summary_dir```

```main.py --debug --start_from_scratch --max_train_epochs 500 --augmentation True --max_aug 5 --grayscale False --resnet 34 --lr 10e-3 --max_pool 1 --conv_kernel_size 9 --batch_size 8 --epochs_per_eval 2 --num_workers 6 --early_stopping_patience_epochs 24 --split 0.7 --across_split True --interval 5 --num_classes 2 --img_size 512 --data_dir image_data_directory --data_split_dir data_split_directory --create_datasets True --model_dir model_storage_directory --log_dir log_dir --checkpoint_dir checkpoint_dir --summary_dir summary_dir```

After successful training procedure the final model will be saved and stored as .pk (pickle) file within the summary folder.


## Network Testing and Evaluation
During training VVI-DETECT will be verified on an independent validation set. In addition VVI-DETECT will be automatically evaluated on the test set at the very end of the training procedure. In both cases multiple machine learning metrics (overall accuracy, recall, precision, and F1-socre, besides class-specific recall, precision, and F1-score) will be calculated and documented. Besides the traditional logging, all scalar, but also visual results (images), and the entire training process is documented and can be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir summary_directory/```

There exist also the possibility to evaluate the VVI-DETECT model on an entire unseen portion of images. The prediction script (*predict.py*) implements a data loading procedure of a given image set, and applies the trained model (loading the model from as .pk file) in order to distinguish between valid and invalid sub-images for subsequent animal-specific individual classification/recognition. Each image will be classified, presenting the predicted class, label, and a posteriori probability, next to an optional given threshold (passed or not passed).

<ins>**Example Command:**</ins>

```predict.py --debug --model_path path_to_folder/model.pk --output_path output_directory --labels_info_file path_to_folder/label_dictionary_X_intervalX.json --log_dir log_dir --image_input_folder image_directory --batch_size 1 --img_size 512 --threshold 0.90 ```

# FIN-IDENTIFY

## Python, Python Libraries, and Version
FIN-IDENTIFY is a deep learning algorithm which has been implemented in Python (Version=3.8) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.8.1, TorchVision=0.9.1). Moreover it requires the following Python libraries: Pillow, Pandas, TensorboardX, Matplotlib, next to a portion of Python libraries already integrated within the standard Python repository (recent versions).

## Required Filename Structure for Training
In order to properly load and preprocess your own animal-specific data to train a network in order to distinguish between different animal individuals based on their natural identifiers/markings, previously detected and equally resized/rescaled target animal-specific sub-images of interest, the filename structure has to follow the original image-specific output structure of FIN-EXTRACT (see above). FIN-IDENTIFY requires exactly the same filename structure as VVI-DETECT. In order to train FIN-IDENTIFY the bounding box (.txt) files are not needed. However, the filename structure of the sub-images has to follow the above shown template (same as the output of FIN-EXTRACT/VVI-DETECT), to ensure a correct training procedure.

## Required Directory Structure for Training
FIN-IDENTIFY is capable of performing its own training, validation, and test split if there is not an existing one. The system either checks whether train.csv, val.csv, and test.csv files already present, or the option *--create_datasets* is set. Otherwise an existing data split has to exist. Using the option *--data_split_dir* specifies the location of the data partitions. In order to generate a proper data split for individual-specifc multi-class classification the following folder structure has to be ensured:

Data_Folder (e.g. containing all the sub-folders with the images)
 
    ├── ANIMAL1 (name of the directory represents the "Ground Truth Label" for all images within that folder)
    │   ├── KWT_2017-02-26_MMalleson_JuandeFuca_038_Fin_cropped_0_2017-02-26_14-15-04.JPG
    │   ├── KWT_2017-02-26_MMalleson_JuandeFuca_063_Fin_cropped_0_2017-02-26_15-56-34.JPG
    │   └── KWT_2017-03-12_MMalleson_RacePassage_036_Fin_cropped_1_2017-03-12_13-37-44.JPG
    │   └── (...)
    ├── ANIMAL2 (name of the directory represents the "Ground Truth Label" for all images within that folder)
    |   ├──  KWT_2014-08-04_DMatkin_BartlettRiverSEAK_016_Fin_cropped_3_2014-08-04_23-20-06.JPG
 	|   ├──  KWT_2014-08-21_DMatkin_GustavusFlatsSEAK_007_Fin_cropped_0_2014-08-21_13-43-45.JPG
 	|   ├──  KWT_2014-09-06_JTowers_ChathamPoint_5918_Fin_cropped_0_2014-09-06_13-22-57.JPG
 	|   ├──  KWT_2014-10-26_JTowers_BlackfishSound_8536_Fin_cropped_2_2014-10-26_15-26-20.JPG
 	|   ├──  KWT_2014-10-26_JTowers_BlackfishSound_8678_Fin_cropped_1_2014-10-26_17-23-12.JPG
 	|   ├──  (...)
    ├── ANIMAL3 (name of the directory represents the "Ground Truth Label" for all images within that folder)
    │   ├── KWT_2017-02-26_MMalleson_JuandeFuca_038_Fin_cropped_0_2017-02-26_14-15-04.JPG
    │   ├── KWT_2017-02-26_MMalleson_JuandeFuca_063_Fin_cropped_0_2017-02-26_15-56-34.JPG
    │   └── KWT_2017-03-12_MMalleson_RacePassage_036_Fin_cropped_1_2017-03-12_13-37-44.JPG
    │   └── (...)
    ├── ANIMAL4 (name of the directory represents the "Ground Truth Label" for all images within that folder)
    |   ├──  KWT_2014-08-04_DMatkin_BartlettRiverSEAK_016_Fin_cropped_3_2014-08-04_23-20-06.JPG
 	|   ├──  KWT_2014-08-21_DMatkin_GustavusFlatsSEAK_007_Fin_cropped_0_2014-08-21_13-43-45.JPG
 	|   ├──  KWT_2014-09-06_JTowers_ChathamPoint_5918_Fin_cropped_0_2014-09-06_13-22-57.JPG
 	|   ├──  KWT_2014-10-26_JTowers_BlackfishSound_8536_Fin_cropped_2_2014-10-26_15-26-20.JPG
 	|   ├──  KWT_2014-10-26_JTowers_BlackfishSound_8678_Fin_cropped_1_2014-10-26_17-23-12.JPG
 	|   ├──  (...)

<ins>**Example of train.csv, valid.csv, and test.csv structure:**</ins>

```
Label,Filepath\
ANIMAL1,path_to_folder/ANIMAL1/KWT_2012-07-09_CMcMillan_PlumperIslands_3109_Fin_cropped_0_2012-07-09_11-53-19.JPG\
ANIMAL2,path_to_folder/ANIMAL2/KWT_2011-03-28_MMalleson_SwansonChannel_0915_Fin_cropped_0_2011-03-28_14-18-14.JPG\
ANIMAL3,path_to_folder/ANIMAL3/KWT_2011-10-06_JTowers_BroughtonStrait_5639_Fin_cropped_0_2011-10-06_13-51-37.JPG\
ANIMAL1,path_to_folder/ANIMAL1/KWT_2015-10-25_MMalleson_Sooke_022_Fin_cropped_0_2015-10-25_16-06-55.JPG\
ANIMAL1,path_to_folder/ANIMAL1/KWT_2012-08-27_MMalleson_MuirCreek_041_Fin_cropped_0_2012-08-27_09-59-10.JPG\
ANIMAL3,path_to_folder/ANIMAL3/KWT_2014-07-18_DMatkin_YoungIslandSEAK_014_Fin_cropped_0_2014-07-18_14-28-31.JPG\
ANIMAL3,path_to_folder/ANIMAL3/KWT_2013-06-23_HTom_PloverReefs_022_Fin_cropped_0_2013-06-23_16-18-44.JPG\
ANIMAL2,path_to_folder/ANIMAL2/KWT_2015-02-23_JTowers_RoundIsland_1656_Fin_cropped_0_2015-02-23_13-43-37.JPG\
ANIMAL4,path_to_folder/ANIMAL4/KWT_2013-10-11_RAbernethy_SquallyChannel_0754_Fin_cropped_0_2013-10-11_12-57-25.JPG\
ANIMAL4,path_to_folder/ANIMAL4/KWT_2011-09-17_JTowers_DonegalHead_4662_Fin_cropped_0_2011-09-17_11-47-52.JPG\
ANIMAL2,path_to_folder/ANIMAL2/KWT_2011-04-15_GEllis_SansumNarrows_3487_Fin_cropped_0_2011-04-15_15-35-08.JPG
```
## Network Training
For a detailed description about each possible training option we refer to the usage/code in *main.py* (usage: *main.py -h*), together with the paper descriptions. 

This is just an example command in order to start network training with an existing data split and without an existing data split:

```main.py --debug --max_train_epochs 500 --augmentation True --max_aug 5 --grayscale False --resnet 34 --lr 10e-3 --max_pool 1 --conv_kernel_size 9 --batch_size 8 --epochs_per_eval 2 --num_workers 6 --early_stopping_patience_epochs 20 --split 0.7 --across_split True --interval 5 --num_classes 101 --img_size 512 --data_split_dir data_split_directory --data_dir image_data_directory --model_dir model_storage_directory --log_dir log_dir --checkpoint_dir checkpoint_dir --summary_dir summary_dir```

```main.py --debug --max_train_epochs 500 --augmentation True --max_aug 5 --grayscale False --resnet 34 --lr 10e-3 --max_pool 1 --conv_kernel_size 9 --batch_size 8 --epochs_per_eval 2 --num_workers 6 --create_datasets True --early_stopping_patience_epochs 20 --split 0.7 --across_split True --interval 5 --num_classes 101 --img_size 512 --data_split_dir data_split_directory --data_dir image_data_directory --model_dir model_storage_directory --log_dir log_dir --checkpoint_dir checkpoint_dir --summary_dir summary_dir```

After successful training procedure the final model will be saved and stored as .pk (pickle) file within the summary folder.

## Network Testing and Evaluation
During training FIN-IDENTIFY will be verified on an independent validation set. In addition FIN-IDENTIFY will be automatically evaluated on the test set at the very end of the training procedure. In both cases multiple machine learning metrics (overall accuracy, topN weighted and unweighted accuracy, besides class-specific recall, precision, and F1-score) will be calculated and documented. Besides the traditional logging, all scalar, but also visual results (images), and the entire training process is documented and can be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir summary_directory/```

There exist also the possibility to evaluate the FIN-IDENTIFY model on an entire unseen portion of images. The prediction script (predict.py) implements a data loading procedure of a given image set, and applies the trained model (loading the model from as .pk file) in order to distinguish between the individual-specific animals of the respective target species. Each image will be classified, presenting the predicted class, label, and a posteriori probability, next to an optional given threshold (passed or not passed). Moreover, the a posteriori probabilities, next to the label information, of the top-K ranked candidates (see option *--topK*) is listed.  

<ins>**Example Command:**</ins>

```predict.py --debug --model_path path_to_folder/model.pk --output_path output_directory --labels_info_file path_to_folder/label_dictionary_X_intervalX.json --log_dir log_dir --image_input_folder image_directory --topK 3 --batch_size 1 --img_size 512 --threshold 0.90 ```