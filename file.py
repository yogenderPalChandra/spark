

#Import what is needed
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SQLContext, Row
from sparknlp.base import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession

import pandas as pd
from bs4 import BeautifulSoup
from pyspark.sql.types import *
from pyspark.sql.functions import udf

from sparknlp.annotator import *
from pyspark.ml import Pipeline
#from py4j.java_gateway import java_import
import sparknlp
import os
sc = sparknlp.start()
#sc.stop()

sqlContext = SQLContext(sc)
path = "./*html"

def rdd_l(path):
    """path to rdd builder
    takes path as argument and returns rdd 
    """

    return sc.sparkContext.wholeTextFiles("./*.html")


def df(rdd_l):
    """ rdd to df builder
    takes list of rdd (rdd_l) as argument and returns rdd dataframe (df).
    rdd stores values as a tuple of filnename and the actual value (HTML doc) in this case
    """
    return rdd_l.toDF(schema=["filename", "text"]).select("text")

import os
def file_for_regex_transformer():
    """Regex rules to file in current directory
    This function returns path to file with rules string for Regex matcher pipline
    in the function nlp_pipline_and_clean(rdd_df)
    """
    rules = '''.\d\&\w+\;\d+&\w+;\d+&\w+;Kč*'''
    with open('regex_rules.txt', 'w') as f:
        f.write(rules)
    return os.path.join(os.getcwd(), "regex_rules.txt")


RegexMatcher().extractParamMap()

def nlp_pipline_and_clean(rdd_df):
    """takes rdd dataframe rdd_df and returns regex matches item i.e. class="norm-price ng-binding
    :DocumentAssembler():  is a sparknlp.base class Transformer 
    which takes rdd with input column (setInputCol()) ->test and returns rdd with column 'assembled'
    :RegexMatcher(): is a the Spark NLP transformer which actually does regex matching 
    of the string we defined in the previous function.  It takes 'assembled' column as input and returns
    'regex_matches' column 
    :nlpPipeline: a pipline is initialised and called on the rdd_df
    """
    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("assembled")

    regex_matcher = RegexMatcher()\
        .setInputCols('assembled')\
        .setStrategy("MATCH_ALL")\
        .setOutputCol("regex_matches")\
        .setExternalRules(path=file_for_regex_transformer(), delimiter=',')

    nlpPipeline = Pipeline(stages=[
        documentAssembler,
        regex_matcher
     ])
    return nlpPipeline.fit(rdd_df).transform(rdd_df) \
.select("regex_matches.result") \
.rdd.flatMap(lambda x: x[0])\
.map(lambda s: s.replace('&nbsp;', '')) \
.map(lambda s: s.replace('>', '')) \
.map(lambda s: s.replace('>', '')) \
.map(lambda s: s.replace('Kč', ' Kč')) \
.map(lambda s: s.split()) \
.map(lambda s: '{:,} {}'.format(int(s[0]), str(s[1])))


def rdd_df_2_pd_df(rdd_df):
    """ changes the name of rdd rows to more identifiable name - cost
    returns pandas df
    """
    row = Row("cost")
    match_df_flat1_df= rdd_df.map(row).toDF()
    return  match_df_flat1_df.toPandas()
    
def rdd_df_2_pd_df(rdd_df):
    """ changes the name of rdd rows to more identifiable name - cost
    returns pandas df
    """
    row = Row("cost")
    match_df_flat1_df= rdd_df.map(row).toDF()
    return  match_df_flat1_df.toPandas()

def rdd_2_address(rdd_df):

    '''Takes rdd dataframe cleans it and returns pandas df 
    :rdd_l.map(lambda x: x[1]): takes the data out of tuple of rdd. because rdd when read using WholeTextFile stores
    tuple(fileNamle, StringData).
    :.map(lambda x: x.split('\n\t')): split the data by combination of next line and tab seperater.
    :.map(lambda x: x[6]): Pick up the 6th row from rdds as it is the intended class="location-text ng-binding"
    :.map(lambda x: x.split('prodeji')): split by prodej to get meaningfull string
    :.map(lambda x: x[1]): some indexing to pick elemnt out of list of list
    :.map(lambda x: x.split(';')): split it at ; because thats where address seperates 
    :.map(lambda x: x[0]): again pick element from lol (list of list) or tuple
    :.map(lambda x: x.strip(' ')): strip out empty space
    '''
    rdd1 = rdd_df.map(lambda x: x[1]).map(lambda x: x.split('\n\t')).map(lambda x: x[6]) \
    .map(lambda x: x.split('prodeji')).map(lambda x: x[1]).map(lambda x: x.split(';')) \
    .map(lambda x: x[0]).map(lambda x: x.strip(' ')).map(lambda x: str(x)).map(lambda p: Row(p))

    schemaString = "address"
    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    schema = StructType(fields)
    addressSchema = sqlContext.createDataFrame(rdd1, schema)
    return addressSchema.toPandas()

def remove_fileName_rdd(rdd_l, columnName=None):
    """Takes rdd_l and change the name according to what we provide
    Also, indexes out the data.
    """
    from pyspark.sql import Row
    columnName = str(columnName)
    row = Row(columnName) # Or some other column name
    return rdd_l.map(lambda x: x[1]).map(row).toDF()



def processRecord_udf_param1(rdd_l):
    """ Beautiful Soup UDF takes rdd_l parses  needed HTML <tag>:class="params1"
    """
    soup = BeautifulSoup(rdd_l, "html.parser")
    classes = []
    for element in soup.find_all('ul', class_='params1'):
        for il in element.find_all('li'):
            text = il.get_text(strip=True).replace(u'\xa0', u' ')
            classes.append(text)

    return classes

def register_apply_udf(rdd_l):
    """ this function applies the UDF UDF to the rdd_l
    """
    apply2lambdafunc = lambda z: processRecord_udf_param1(z)

    return remove_fileName_rdd(rdd_l(path), columnName='val'). \
withColumn('cleaned', udf(apply2lambdafunc, StringType())('val')) \
.select("cleaned").rdd.flatMap(lambda x: x)

def cleaned2pd_param1(cleaned_rdd_l):
    """convert the parsed information i.e. parama1 to rdd df and then to Pandas df
    """
    pd.set_option('display.max_colwidth', None)
    row = Row("cleanedParams1") 
    return cleaned_rdd_l.map(row).toDF().select('cleanedParams1').toPandas()


def processRecord_udf_param2(file):
    """ Same function but for Param2 parsing 
    using Beautiful Soup UDF takes class='params2' <tag>
    """
    soup = BeautifulSoup(file, "html.parser")

    classes = []
    for element in soup.find_all('ul', class_='params2'):
        for il in element.find_all('li'):
            text = il.get_text(strip=True).replace(u'\xa0', u' ')

            classes.append(text)
    return classes

def register_apply_udf_param2(rdd_l):
    """same apply UDF to rdd_l
    """
    apply2lambdafunc = lambda z: processRecord_udf_param2(z)
    return remove_fileName_rdd(rdd_l(path), columnName='val'). \
withColumn('cleaned', udf(apply2lambdafunc, StringType())('val')) \
.select("cleaned").rdd.flatMap(lambda x: x)


def cleaned2pd_param2(cleaned_rdd_l):
    """cleaned tags to df to pandas df
    """
    pd.set_option('display.max_colwidth', None)
    row = Row("cleanedParams2") 
    return cleaned_rdd_l.map(row).toDF().select('cleanedParams2').toPandas()

def pd_2_csv(df1, df2, df3, df4):
    """concat all pandas dfs. 
    One can choose to join rdd dfs itself but I find pandas more easy to work with.
    Pandas work with driver node only, so it is a layer which reduces data on driver.
    from this step ownwards, computing is mo more distributed. To solve this issue one can even use
    pandas pyspark.pandas library: https://spark.apache.org/
    """
    df_concat = pd.concat ([df1, df2, df3, df4], axis=1)
    df_concat.to_csv("cleaned_data.csv", sep=',')
    return df_concat

if __name__ == "__main__":

    pd_2_csv(rdd_df_2_pd_df(nlp_pipline_and_clean(df(rdd_l(path)))), \
        rdd_2_address(rdd_l(path)),\
        cleaned2pd_param1(register_apply_udf(rdd_l)),\
        cleaned2pd_param2(register_apply_udf_param2(rdd_l)))
        
print ("done, check cleaned_data.csv")
