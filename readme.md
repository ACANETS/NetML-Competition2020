# NetML Challenge 2020

## Update as of January 14, 2022:

**After a long delay from the sponsor's approval process (legal , finance team, etc) due to COVID, staff shortage and policy clarification, we are informed that the approval process is finished. The next step on the sponsor side is to place gift orders within respective geographical regions, where the awardees are located. We are hopeful that the awards will arrive soon. We sincerely thank you for your patience during this unprecedented time.**

## Update as of November 24, 2021
We wanted to let the NetML awardees know that our sponsor Intel is still going through their internal process to issue the awards. The complexity lies in that the awardees are from different geographical regions, which have to be handled under different governing laws and policies. We have asked Intel to expedite and they are working on that. We appreciate your patience.

## Update as of July 21, 2021
Dear Competition Participants,
Since we announced the results of the Competition in March 2021, we have been in close contact with Intel Corporation, the sponsor of the competition to figure out the logistics of making the awards to the awardees who are from world-wide. Due to the COVID-19 pandemic and unforeseen complexity (e.g., the difference in rules and policies specific to geographic regions), the process of distributing the awards to recipients is significantly delayed. Although this is beyond our control unfortunately, we are diligently working with relevant Intel divisions and teams to resolve this as soon as we can.

We again appreciate your patience. Please do not hesitate to email us (Yan_Luo@uml.edu and Onur_Barut@student.uml.edu) if you have additional questions.

Thank you.

NetML competition organizers:

Yan Luo, UMass Lowell

Tong Zhang, Intel Corporation

Onur Barut, UMass Lowell

Peilong Li, Elizabethtown College

----------------

## Overview
Recent progress in AI, Machine Learning and Deep Learning has demonstrated tremendous success in many application domains such as games and computer vision. Meanwhile, there are challenges of proliferating data flows and increasing malicious traffic on today’s Internet that call for advanced network traffic analysis tools. In this competition, we challenge the participants to leverage novel machine learning technologies to detect malicious flows and/or distinguish applications in a fine-grained fashion among network flows.

We are pleased to announce the NetML Challenge 2020. Given a set of flow features extracted from the original packet capture (PCAP) files, the task is to provide an accurate prediction. The challenge is hosted on EvalAI. Challenge link: https://evalai.cloudcv.org/web/challenges/challenge-page/526/overview

This NetML Challenge 2020 is the 1st of the Machine Learning Driven Network Traffic Analytics Challenge. In this year’s challenge, a collection of 1,199,139 flows in three different datasets are given including detailed flow features and labels. We also provide simple APIs and baseline machine learning models to demonstrate the usage of the datasets and evaluation metrics. There are 7 tracks for specific analytics objectives.

Two different tasks, namely malware detection and traffic classification, are available in this challenge. We provide two separate malware detection datasets, namely NetML and CICIDS2017, and one dataset for traffic classification: non-vpn2016.

- NetML dataset is constructed by selecting several PCAP files from www.stratosphereips.org website.
- CICIDS2017 dataset is generated using https://www.unb.ca/cic/datasets/ids-2017.html
- non-vpn2016 dataset is the subset of ISCX-VPN-nonVPN2016 dataset from https://www.unb.ca/cic/datasets/vpn.html

Data for each dataset are available under the data folder in this repository. For NetML and CICIDS2017 datasets, two different annotations are available: top-level and fine-grained. Top-level annotations are for binary classification, i.e. benign or malware. For fine-grained, each dataset has different malware types and participants are required to label each flow according to their malware classes. For non-vpn2016, three level of annotations are available: top-level, mid-level and fine-grained. Annotations on the training sets are publicly available.

This challenge is organized by the Laboratory of Advanced Computer Architecture and Network Systems (ACANETS) at the University of Massachusetts Lowell, and sponsored by Intel Corporation. The results will be announced at NetML Challenge Workshop in IJCAI 2020.

----------------

## Reference
For detailed description about the datasets please check our paper https://arxiv.org/abs/2004.13006

If you find this study useful, please cite our paper:
```
@article{DBLP:journals/corr/abs-2004-13006,
  author    = {Onur Barut and
               Yan Luo and
               Tong Zhang and
               Weigang Li and
               Peilong Li},
  title     = {NetML: {A} Challenge for Network Traffic Analytics},
  journal   = {CoRR},
  volume    = {abs/2004.13006},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.13006},
  eprinttype = {arXiv},
  eprint    = {2004.13006},
  timestamp = {Tue, 16 Feb 2021 15:53:52 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-13006.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

----------------

## Dates
- February 17, 2020               NetML challenge is launched!
- *<del>June 1, 2020                    Submission deadline at 23:59:59 UTC</del>*
- *<del>June 15, 2020                   Winners' announcement at the NetML Workshop, IJCAI 2020</del>*
- **COVID-19 UPDATE !!! Challenge submission deadline and date of winner's announcement are postponed to comply with IJCAI 2020. You may follow https://ijcai20.org/ for the conference and https://netaml.github.io/ for the workshop updates.**
- December 1, 2020                    Submission deadline at 23:59:59 UTC
- January 7, 2021                     Winners' announcement at the NETAML Workshop, IJCAI 2020
----------------

## Prize
$3000 prize will be awarded for the winners in each of seven tracks. The total amount is going to be split as shown below:

| Tracks  |    Winner   | Runner-up | 2nd Runner-up |
| ------- | ----------- | --------- | ------------- |
|   #1    |     $150    |    $125   |     $100      |
|   #2    |     $200    |    $150   |     $100      |
|   #3    |     $150    |    $125   |     $100      |
|   #4    |     $200    |    $150   |     $100      |
|   #5    |     $200    |    $150   |     $100      |
|   #6    |     $200    |    $150   |     $100      |
|   #7    |     $200    |    $150   |     $100      |

----------------

## Challenge Guidelines

We have divided each dataset into 3 splits, test-challenge set, test-std set, and training set. Test-challenge set and test-std set are constructed by randomly extracting 10% of each class under fine-grained labels and the remaining 80% is left as training set. Annotations for training set is publicly available, but annotations for test-std and test-challenge tests will be private.

There are 7 tracks, and each track has two phases such as *dev* and *challenge*, making 14 different phases. First 7 phases are for development and last 7 phases are for the challenge. Development phase leaderboard will be public for participants to compare their team with other while challenge phase leaderboard will be announced after the challenge deadline. Below we describe each phase.

### 1- NetML_toplevel_dev
You will use NetML training dataset to train you own model using top-level annotations (benign or malware) and submit your predictions on the NetML test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 2- NetML_finegrained_dev
You will use NetML training dataset to train you own model using fine-grained annotations (benign, adload, ransomware, etc..) and submit your predictions on the NetML test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 3- CICIDS2017_toplevel_dev
You will use CICIDS2017 training dataset to train you own model using top-level annotations (benign or malware) and submit your predictions on the CICIDS2017 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 4- CICIDS2017_finegrained_dev
You will use CICIDS2017 training dataset to train you own model using fine-grained annotations (benign, webAttack, portScan, etc..) and submit your predictions on the CICIDS2017 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 5- non-vpn2016_toplevel_dev
You will use non-vpn2016 training dataset to train you own model using top-level annotations (chat, audio, video, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 6- non-vpn2016_midlevel_dev
You will use non-vpn2016 training dataset to train you own model using mid-level annotations (facebook, skype, hangouts, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 7- non-vpn2016_finegrained_dev
You will use non-vpn2016 training dataset to train you own model using fine-grained annotations (facebook_audio, facebook_chat, skype_audio, skype_chat, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 8- NetML_toplevel_challenge
You are expected to use top-level annotations (benign or malware) and submit your predictions on the NetML test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 9- NetML_finegrained_challenge
You are expected to use fine_grained annotations (benign, adload, ransomware, etc..) and submit your predictions on the NetML test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 10- CICIDS2017_toplevel_challenge
You are expected to use top-level annotations (benign or malware) and submit your predictions on the CICIDS2017 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 11- CICIDS2017_finegrained_challenge
You are expected to use fine-grained annotations (benign, webAttack, portScan, etc..) and submit your predictions on the CICIDS2017 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 12- non-vpn2016_toplevel_challenge
You are expected to use top-level annotations (chat, audio, video, etc..) and submit your predictions on the non-vpn2016 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 13- non-vpn2016_midlevel_challenge
You are expected to use mid-level annotations (facebook, skype, hangouts, etc..) and submit your predictions on the non-vpn2016 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 14- non-vpn2016_finegrained_challenge
You are expected to use fine-grained annotations (facebook_audio, facebook_chat, skype_audio, skype_chat, etc..) and submit your predictions on the non-vpn2016 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

----------------

Three prize winners (The Winner, Runner-up, 2nd Runner-up) will be announced for each track. In order to win the prize, for example for non-vpn2016 toplevel track, a team must submit their results for both non-vpn2016_toplevel_dev and vpn2016_toplevel_challenge phases. The final score for each track will be calculated by averaging the 'overall' scores of both *dev* and *challenge* phases.

The evaluation page lists detailed information regarding how submissions will be scored. The evaluation servers are open. To enter the competition, first you need to create an account on EvalAI, developed by the CloudCV team. EvalAI is an open-source web platform designed for organizing and participating in challenges to push the state of the art on AI tasks.

To submit your JSON file to the NetML evaluation servers, click on the “Submit” tab on the NetML challenge 2020 page on EvalAI. Select the phase. Please select the JSON file to upload and fill in the required fields such as "method name" and "method description" and click “Submit”. After the file is uploaded, the evaluation server will begin processing. To view the status of your submission please go to “My Submissions” tab and choose the phase to which the results file was uploaded. If the status of your submission is “Failed” please check your "Stderr File" for the corresponding submission.

After evaluation is complete and the server shows a status of “Finished”, you will have the option to download your evaluation results by selecting “Result File” for the corresponding submission. The "Result File" will contain the aggregated accuracy on the corresponding data split.

Please limit the number of entries to the challenge evaluation server to a reasonable number, e.g., one entry per paper. To avoid overfitting, the number of submissions per user is limited to 1 upload per day (according to UTC timezone) and a maximum of 5 submissions per user. It is not acceptable to create multiple accounts for a single project to circumvent this limit. The exception to this is if a group publishes two papers describing unrelated methods, in this case both sets of results can be submitted for evaluation. However, Development phases allow for 10 submissions per day.

The download page contains links to all NetML, CICIDS2017 and non-vpn2016 train, test-std and test-challenge data and associated annotations (for training set only). Please specify any and all external data used for training in the "method description" when uploading results to the evaluation server.

Results must be submitted to the evaluation server by the challenge deadline. Competitors' algorithms will be evaluated according to the rules described on the evaluation page. Challenge participants with the most successful and innovative methods will be invited to present.

----------------

## Dataset Description
Three different dataset are provided in this challenge: NetML, CICIDS2017, non-vpn2016. A proprietary flow feature extraction library is utilized to extract several META, TLS, DNS and HTTP flow features from raw traffic. These features are described in flow_features.csv file.
 
The 10% of each class is allocated to test-challenge set, another 10% for test-std set, and the remaining 80% of the extracted flow feature set is kept for the traininig set which are stored in the three files, 0_test-challenge_set.json.gz, 1_test-std_set.json.gz, 2_training_set.json.gz. Every flow for each dataset has been assigned a unique flow id number. Every flow for NetML and CICIDS2017 datasets has two different annotations, top-level and fine-grained. Every flow in non-vpn2016 dataset has three different annotations, top-level, mid-level, fine-grained. Annotations, i.e. ground-truth are separately stored for each level of annotations.


### NetML
Raw traffic (e.g., Pcap files) is obtained from www.stratosphereips.org. This dataset has 20 types of attacks and normal traffic flows. 

The total number of flows for different splits:
- test-challenge set: 48,394 
- test-std set : 48,394
- traininig set: 387,268

### CICIDS2017
Raw traffic is obtained from https://www.unb.ca/cic/datasets/ids-2017.html. Attack flows are extracted by filtering each workday PCAP files with respect to time interval and IPs described in their webpage. The extracted dataset has 7 types of malware attacks and normal traffic flows.

The total number of flows for different splits:
- test-challenge set: 55,128 
- test-std set : 55,128
- traininig set: 441,116

### non-vpn2016
PCAP files are downloaded from https://www.unb.ca/cic/datasets/vpn.html. The original dataset has both vpn and non-vpn packet capture files but we only focus on non-vpn captures. In top-level annotation, we categorize the traffic into 7 groups: audio, chat, email, file_transfer, tor, video, P2P. In mid-level annotation, we group into 18 classes according to the application type such as aim_chat, facebook, hangouts, skype, youtube etc. In fine-level annotation, we treat each action as a different category and obtain 31 classes such as facebook_chat, facebook_video, skype_chat, skype_video etc.

The total number of flows for different splits:
- test-challenge set: 16,323 
- test-std set : 16,323
- traininig set: 131,065

----------------

## Quickstart Guideline

Train a simple Random Forest model using sklearn and create submission.json file.

### Requirements
- python 3
- numpy
- pandas
- sklearn
- matplotlib

We recommend you to use virtual environment. For details please see https://docs.python.org/3/tutorial/venv.html

```shell
$ git clone https://github.com/ACANETS/NetML-Competition2020
```

### Usage Example
Provided baseline script evaluates the model according to the validation set scores created using 20% of the training set. The remaining 80% of the training set is used to train the model.

To train for Phase-1 NetML_toplevel_dev and create a JSON file ready to submit:

```shell
$ python3 RF_baseline.py --dataset ./data/NetML --anno top --submit test-std
```

The above command runs the python script and creates a './results' folder in the main directory. You can find your models confusion matrix with F1 and mAP scores and submission_test.json file under the folder with the time you ran the above command.

This script uses the functions defined under ./utils/helper.py to read the data, display and save the confusion matrix, and create the submission json file.

- Load data and split the validation set:

```python
# Get training data in np.array format
Xtrain, ytrain, class_label_pair, Xtrain_ids = get_training_data(training_set, training_anno_file)

# Split validation set from training data
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=ytrain)
```

The above function uses ./utils/featureDict_META.json file to parse the META features from json.gz files but you are encouraged to write your own parser to include other features as well!

- Preprocess for a better performance:

```python
# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

- Define and train the model. Print training and validation set accuracy:

```python
# Train RF Model
print("Training the model ...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
clf.fit(X_train_scaled, y_train)

# Output accuracy of classifier
print("Training Score: ", clf.score(X_train_scaled, y_train))
print("Validation Score: ", clf.score(X_val_scaled, y_val))
```

- Plot the confusion matrix and save under ./results/<%Y%m%d-%H%M%S>/CM.png:

```python
# Print Confusion Matrix
ypred = clf.predict(X_val_scaled)

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred, 
                        classes=class_names, 
                        normalize=False)
```

- If you want to create a submission json file for a selected set, the following script handles it and saves the selected json under ./results/<%Y%m%d-%H%M%S>/ folder:

```python
# Make submission with JSON format
if args.submit == "test-std" or args.submit == "both":
    submit(clf, test_set, scaler, class_label_pair, save_dir+"/submission_test-std.json")
if args.submit == "test-challenge" or args.submit == "both"
    submit(clf, challenge_set, scaler, class_label_pair, save_dir+"/submission_test-challenge.json")
```

----------------

### Evaluation
Submission file should be as the following JSON format:

```python
{"id1": "label1", "id2": "label2", ...}
```

For multi-class classification problems, we use F1 score and mean average precision (mAP) as two different metrics.

F1=2\*precision\*recall/(precision+recall)

where:

- precision=TP/(TP+FP)
- recall=TP/(TP+FN)

- True Positive [TP] = your prediction is 1, and the ground truth is also 1 - you predicted a positive and that's true!
- False Positive [FP] = your prediction is 1, and the ground truth is 0 - you predicted a positive, and that's false.
- False Negative [FN] = your prediction is 0, and the ground truth is 1 - you predicted a negative, and that's false.

mAP = 1/N\*(\sum(AP<sub>i</sub>))

where:

- N: number of classes
- AP: average precision for each class

'overall' score is bounded between 0 and 1 and higher value represents more accurate results. It is calculated by multiplying the two scores: 

overall = F1\*mAP

----------------

For binary classifications, in other words detection problems, we use True Positive Rate (TPR) as detection rate of malware and False Alarm Rate as two metrics.

- TPR = TP/(TP+FN)
- FAR = FP/(TN+FP)

'overall' score is bounded between 0 and 1 and higher value represents more accurate results. It is calculated by multiplying detection rate with the 1-FAR: 

overall = TPR\*(1-FAR)

----------------

## Directory Structure
* NetML-Competition2020
    - readme.md
    - RF_baseline.py
    - NetML-competition.ipynb
    - flow_features.csv
    * data
        - labels.txt
        * CICIDS2017
            - ...
        * NetML
            - ...
        * non-vpn2016
            - ...
    * results
        * CICIDS2017_fine
            - CM.png
        * CICIDS2017_top
            - CM.png
        * NetML_fine
            - CM.png
        * NetML_top
            - CM.png
        * non-vpn2016_fine
            - CM.png
        * non-vpn2016_mid
            - CM.png
        * non-vpn2016_top
            - CM.png
    * utils
        - featureDict_META.json
        - helper.py

----------------

## License

BSD
