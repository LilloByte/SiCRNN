# SiCRNN: a Siamese Approach for Sleep Apnea Identification via Tracheal Microphone Signals Official Implementation.
This repository contains the scripts for the implementation of the paper SiCRNN: a Siamese Approach for Sleep Apnea Identification via Tracheal Microphone Signals – Official Implementation.
Before starting, download the dataset from the following link: https://www.scidb.cn/en/detail?dataSetId=778740145531650048 (We recommend downloading the latest available version of the dataset).
Furthermore, for a correct reproduction of the experiment, please refer to the patients included in the Train, Validation, and Test sets (note that not all patients in the dataset were used).

Below is a summary table of the patients included in the various datasets used.

### Summary table of patient IDs used for each data split divided into training, validation and test set

| Split      | Train                                                                                                                                     | Validation                         | Test                                         |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|----------------------------------------------|
| **Split 1** | 1112, 1110, 1108, 1106, 1095, 1093,<br>1089, 1088, 1086, 1082, 1071, 1069,<br>1057, 1045, 1041, 1039, 1037, 1028,<br>1022, 1010, 1008, 1006, 995 | 1120, 1104, 1043,<br>1024, 1018, 1014 | 1118, 1116, 1073, 1059,<br>1026, 1020, 1000, 999 |
| **Split 2** | 1120, 1118, 1116, 1112, 1110, 1108,<br>1106, 1104, 1095, 1093, 1089, 1088,<br>1086, 1082, 1073, 1071, 1069, 1059,<br>1057, 1045, 1043, 1041, 1039 | 1010, 1008, 1006,<br>1000, 999, 995 | 1037, 1028, 1026, 1024,<br>1022, 1020, 1018, 1014 |
| **Split 3** | 1086, 1082, 1073, 1071, 1069, 1059,<br>1057, 1045, 1043, 1041, 1039, 1037,<br>1028, 1026, 1024, 1022, 1020, 1018,<br>1014, 1010, 1008, 1006, 1000 | 1120, 1118, 1116,<br>1112, 999, 995 | 1110, 1108, 1106, 1104,<br>1095, 1093, 1089, 1088 |

## Section 1: Dataset Analysis and Preparation



For further information please contact me at: d.lillini@pm.univpm.it
