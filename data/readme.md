# Data for NetML Challenge 2020
This directory contains json.gz files for the three datasets prepared for the NetML Challenge 2020:

- CICIDS2017
- NetML
- non-vpn2016

Each folder contains training_set, test-std_set, and challenge-std_set folder for the data splits. Training_set annotations are also available for supervised training. Test_set and challenge_set annotations will be private until the end of the competition deadline.

labels.txt file contains the name of the classes of each dataset for different level of annotations. As participants of the challenge you shall not modify this file.

## Directory Structure
* data
    - readme.md
    - labels.txt
    * CICIDS2017
        * 0_test-challenge_annotations
            - 0_test-challenge_anno_fine.json.gz
            - 0_test-challenge_anno_mid.json.gz
            - 0_test-challenge_anno_top.json.gz
        * 0_test-challenge_set
            - 0_test-challenge_set.json.gz
        * 1_test-std_annotations
            - 1_test-std_anno_fine.json.gz
            - 1_test-std_anno_mid.json.gz
            - 1_test-std_anno_top.json.gz
        * 1_test-std_set
            - 1_test-std_set.json.gz
        * 2_training_annotations
            - 2_training_anno_fine.json.gz
            - 2_training_anno_mid.json.gz
            - 2_training_anno_top.json.gz
        * 2_training_set
            - 2_training_set.json.gz
    * NetML
        * 0_test-challenge_annotations
            - 0_test-challenge_anno_fine.json.gz
            - 0_test-challenge_anno_mid.json.gz
            - 0_test-challenge_anno_top.json.gz
        * 0_test-challenge_set
            - 0_test-challenge_set.json.gz
        * 1_test-std_annotations
            - 1_test-std_anno_fine.json.gz
            - 1_test-std_anno_mid.json.gz
            - 1_test-std_anno_top.json.gz
        * 1_test-std_set
            - 1_test-std_set.json.gz
        * 2_training_annotations
            - 2_training_anno_fine.json.gz
            - 2_training_anno_mid.json.gz
            - 2_training_anno_top.json.gz
        * 2_training_set
            - 2_training_set.json.gz
    * non-vpn2016
        * 0_test-challenge_annotations
            - 0_test-challenge_anno_fine.json.gz
            - 0_test-challenge_anno_mid.json.gz
            - 0_test-challenge_anno_top.json.gz
        * 0_test-challenge_set
            - 0_challenge_set.json.gz
        * 1_test-std_annotations
            - 1_test-std_anno_fine.json.gz
            - 1_test-std_anno_mid.json.gz
            - 1_test-std_anno_top.json.gz
        * 1_test-std_set
            - 1_test-std_set.json.gz
        * 2_training_annotations
            - 2_training_anno_fine.json.gz
            - 2_training_anno_mid.json.gz
            - 2_training_anno_top.json.gz
        * 2_training_set
            - 2_training_set.json.gz
