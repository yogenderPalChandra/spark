# How to work with Spark UDF and Spark NLP
# Assignment
#### Analytical team requires the following data to be extracted:
#### 1. Price value of class="norm-price ng-binding"
#### 2. Location value of class="location-text ng-binding"
#### 3. All parameters labels and values in class="params1" and class="params2"

*Prepare environment (one can use conda environment):*
  * conda create -n pyspark_env
  * conda activate pyspark_env
  * conda install -c conda-forge pyspark
  
*Export path to where pyspark is installed:*
 * export SPARK_HOME=/path/to/pyspark #Check correct path! its important otherwise spark will not run
 * source ~/.bashrc #optional if you want to launch from other terminals too

*Try if spark runs:*
  * spark-shell #works

*Install spark-nlp #https://nlp.johnsnowlabs.com/docs/en/install*
  * conda install -c johnsnowlabs spark-nlp

*Export python path for both driver and worker nodes (python version should be same for both)*
  * export PYSPARK_PYTHON="path/to/python" # path should be something like /conda/env/bin/...
  * export PYSPARK_DRIVER_PYTHON="path/to/python"

*Install Beautiful Soup too*
  * conda install -c anaconda beautifulsoup4


