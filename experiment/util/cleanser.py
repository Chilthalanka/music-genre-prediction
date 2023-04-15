# Import libraries
from pyspark import keyword_only
from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, trim
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class Cleanser(Transformer,               # Base class
               HasInputCol,               # Sets up an inputCol parameter
               HasOutputCol,              # Sets up an outputCol parameter
               DefaultParamsReadable,     # Makes parameters readable from file
               DefaultParamsWritable      # Makes parameters writable from file
               ):
    """
    Custom transformer wrapper class for removing punctuation marks and double spaces
    """

    inputCol = Param(Params._dummy(),
                     "inputCol",
                     "input column name",
                     typeConverter=TypeConverters.toString)

    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output column name",
                      typeConverter=TypeConverters.toString)


    @keyword_only
    def __init__(self, inputCol=None, outputCol=None) -> None:
        """
        Constructor: set values for all Param objects
        """
        super().__init__()
        self._setDefault(inputCol=None, outputCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)


    @keyword_only
    def setParams(self, inputCol: str = "input", outputCol: str = "output"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def getInputCol(self):
        return self.getOrDefault(self.inputCol)


    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)


    # Required if you use Spark >= 3.0
    def setInputCol(self, new_inputCol):
        return self.setParams(inputCol=new_inputCol)


    # Required if you use Spark >= 3.0
    def setOutputCol(self, new_outputCol):
        return self.setParams(outputCol=new_outputCol)


    def _transform(self, df: DataFrame):
        """
        This is the main member function which applies the transform to
        transform data from the `inputCol` to the `outputCol`
        """
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        single_quote_pattern = "â€™"
        punc_pattern = "[^\w\s]"
        space_pattern = "\s{2,}"

        return df.withColumn(outputCol,
                             trim(regexp_replace(regexp_replace(regexp_replace(inputCol, single_quote_pattern, "\'"),\
                                                                punc_pattern, ""), space_pattern, " ")))
