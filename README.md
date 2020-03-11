# KKStream Deep Learning Workshop
Given the browsing history of the last few months, predict whether the user will use the app in different time slot of the next two weeks.

##### *Detail of the work can be found in Report.pdf

### Environment 
* Python 3.6
* Numpy: 1.15
* Pandas: 0.23.4
* Pytorch: 1.0.1
* Scikit-learn 0.21.1

### Usage
* Download best model
```sh
$ bash download.sh
```
* Preprocedding 
First, set configurations in preprocessing_config.json
```sh
$ python Preprocessing.py
```
* Training 
First, set configuration in training_config.json
```sh
$ python Train.py
```
* Testing
First, set configurations in testing_config.json
```sh
$ python Testing.py
```
