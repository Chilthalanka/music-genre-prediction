# Import libraries
import numpy as np
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class FeatureConcatenator(Transformer,               # Base class
                          HasInputCol,               # Sets up an inputCol parameter
                          HasOutputCol,              # Sets up an outputCol parameter
                          DefaultParamsReadable,     # Makes parameters readable from file
                          DefaultParamsWritable      # Makes parameters writable from file
                          ):
    """
    Custom transformer wrapper class for concatenating TF-IDF and Word2Vec features
    """

    inputCol1 = Param(Params._dummy(),
                      "inputCol1",
                      "first input column name",
                      typeConverter=TypeConverters.toString)

    inputCol2 = Param(Params._dummy(),
                      "inputCol2",
                      "second input column name",
                      typeConverter=TypeConverters.toString)

    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output column name",
                      typeConverter=TypeConverters.toString)


    @keyword_only
    def __init__(self, inputCol1=None, inputCol2=None, outputCol=None) -> None:
        """
        Constructor: set values for all Param objects
        """
        super().__init__()
        self._setDefault(inputCol1=None, inputCol2=None, outputCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, inputCol1: str = "input1",
                  inputCol2: str = "input2",
                  outputCol: str = "output"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def getInputCol1(self):
        return self.getOrDefault(self.inputCol1)

    def getInputCol2(self):
        return self.getOrDefault(self.inputCol2)


    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)


    # Required if you use Spark >= 3.0
    def setInputCol1(self, new_inputCol1):
        return self.setParams(inputCol1=new_inputCol1)

    def setInputCol2(self, new_inputCol2):
        return self.setParams(inputCol2=new_inputCol2)


    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)


    def _transform(self, df: DataFrame):
        """
        This is the main member function which applies the transform
        to transform data from the `inputCol` to the `outputCol`
        """
        inputCol1 = self.getInputCol1()
        inputCol2 = self.getInputCol2()
        outputCol = self.getOutputCol()

        # Concatenate TF-IDF and Word2Vec features
        concatenator_udf = udf(lambda tfidf, w2v: np.concatenate([tfidf, w2v]), ArrayType(DoubleType()))

        return df.withColumn(outputCol, concatenator_udf(inputCol1, inputCol2))
