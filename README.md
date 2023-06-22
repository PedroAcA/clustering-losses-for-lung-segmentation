# Overview
This project performs unsupervised lung segmentation using losses based on intensity clustering at training phase. 

# Model testing
To replicate our results, please follow the steps below.
## Software setup
1. Install Python, Pytorch and Torchvision (a python virtual enviroment can be used)
	- For the GPU version, install the CUDA Toolkit 11.3 available at https://developer.nvidia.com/cuda-11.3.0-download-archive (last access on 20th, June 2023) and run `python -m pip install -r requirements.txt` on a prompt opened in this folder.
		- This project was also tested on the following python, cuda development toolkit, pytorch and torchvision versions 3.10.6, 11.7, 1.13.1+cu117, and 0.14.1+cu117. So, it is expected that versions of these softwares ranging from those tested on cuda developemnt toolkit 11.3 up to 11.7 and their respective python, pytorch and torchvision equivalents should also work. 
	- The usage of pytorch cpu version was not tested, therefore no guarantees about its exectuion or running performance can be made. Should it be needed, edit the requirements.txt for pytorch and torchvision cpu download and run `python -m pip install -r requirements.txt`. 
	
## Data download
1. Download JSRT data from https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt (last access on 19th of June, 2023), uncompress it, convert the X-ray images to jpg format, and move these files into datasets/JSRT_dataset/jpg_imgs/ folder

2. Download SCR masks for JSRT from https://drive.google.com/drive/folders/1FqT2XsBQ_cPwGkq8g1QnSg2PUmh9E9fq?usp=sharing (last access in 19th of June, 2023), uncompress, and move the lungs/ folder into datasets/JSRT_dataset/masks/ folder

3. Download weights from https://drive.google.com/drive/folders/1GkXULPl6bUjAp8wd0KvtwXDcnbvQVoyF?usp=sharing  (last access on 19th of June, 2023) and uncompress it into experiments/adam/epochs_100/ folder. The experiments described are associated to the following folders:
	- RFCM1: RFCM_2_classes_adam/
	- RFCM2: RFCM_2_classes_gamma_0_5_equalize_adam/
	- RFCM3: RFCM_3_classes_adam/ 
	- RFCM4: RFCM_3_classes_gamma_0_5_equalize_adam/
	- MS1: MS_2_classes_adam/
	- MS2: MS_2_classes_gamma_0_5_equalize_adam/
	- MS3: MS_3_classes_adam/ 
	- MS4: MS_3_classes_gamma_0_5_equalize_adam/

4. For results on Montgomery County (MC) dataset, download both the X-rays and masks from https://drive.google.com/drive/folders/15S6r-TxdQEkjK4GvaYxqp2Wuk9E7nJM7?usp=sharing (last access on 21st of JUne, 2023). It has both the right and left lung masks from https://openi.nlm.nih.gov/faq#faq-tb-coll (last access on 20th of June, 2023) merged into a single file. After downloading, uncompress them and move the NLM-MontgomeryCXRSet/ folder to the datasets/ folder

	
## Model running
Note: All the command lines shown below expect that the prompt window is opened inside the src/ folder. 

- If running RFCM2, RFCM4, MS2 or MS4, pre-process the X-Ray input using `python preprocess_input.py --data_root ../datasets/JSRT_dataset/jpg_imgs/ --out_dir ../datasets/JSRT_dataset/pre_processed_2048_2048/` to pre-process images stored on datasets/JSRT_dataset/jpg_imgs/ and save the results into /datasets/JSRT_dataset/pre_processed_2048_2048/

- To run the model, use the following command `python main.py --config_file ../configs/adam/epochs_100/RFCM_3_classes_pre_processed.yaml --mode test --chkpt_file ../experiments/adam/epochs_100/RFCM_3_classes_gamma_0_5_equalize_adam/RFCM_3_classes_gamma_0_5_equalize_adam_epoch_99.pth --test_batch_size 1 --test_type quantitative. This examples run the RFCM4 experiment for quantitative results`. To change the experiments, change the arguments --config_file and --chkpt_file to point to the config and .pth files respectively. To perform qualitative results, change --test_type to qualitative or all (note: qualitative tests need --test_batch_size to be 1 in order to work properly).
		
	- To run MC results of RFCM4, run `python preprocess_input.py --data_root ../datasets/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/ --out_dir ../datasets/NLM-MontgomeryCXRSet/MontgomerySet/pre_processed_mc/` and change --config_file to ../configs/adam/epochs_100/RFCM_3_classes_pre_processed_mc.yaml


# Model training
1. Run `python main.py --config_file ../configs/adam/epochs_100/RFCM_3_classes_pre_processed.yaml --mode train` on command line. This is going to train a new model according to RFCM4 experiment hyperparameters. The .yaml can be edited for different values of hyperparameters or a new .yaml file can be created hyperparameters or a new .yaml file can be created
