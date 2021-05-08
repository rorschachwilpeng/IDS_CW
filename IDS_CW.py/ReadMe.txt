Module: Introduction to Data Science coursework. This project try to analysis freddiemac's dataset for a binary classification problem. Details including building classifications model also final test result visualization.


Dataset: http://www.freddiemac.com/research/datasets/sf_loanlevel_dataset.page



Evaluation Workﬂow ﬁles:

Consists of two subfolders “Classiﬁers” and “Data set and variable correlation analysis”.

Classiﬁers:

This ﬁle includes three models that have been trained, as well as reading test data and calling the trained model to predict base on the test data.All data used for visualization results will be output to the OUTPUT folder(Confusion Matrix and AUC-ROC Curve).

.e.g. for the KNN folder KNN Trained Model.sav —> KNN trained model( Training data: 2015 ~ 2017 dataset)

KNN.py —> the process of training model

Loader.py —> call the trained model to predict the test data

run_ﬁle.ipynb —> A jupyter ﬁle for running the program. You can open it with jupyter directly and the code for running was already included or other way to compile and run “Loader.py”. The output will be written to OUTPUT folder.

The corresponding ﬁles of the other two models are the same operation method.

Data set and variable correlation analysis:

The realization of all the pictures mentioned in the report, “run_ﬁle.ipynb” is for running. The output will be included in “OUTPUT” folder.

Final Model Workﬂow:

The operation method is the same as the model described above

Original Data File Test Datasets