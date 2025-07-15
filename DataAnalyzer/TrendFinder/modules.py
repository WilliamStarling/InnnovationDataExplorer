#For setting up and creating the DSPy signatures and modules needed to search for trends as a part of the bigger data analysis agent.
import dspy
from attachments.dspy import Attachments

class doc_analyzer_sql(dspy.Signature):
  """You will receive a single document from a collection, and using the given categories you will generate a row of csv data that helps summarize the document."""
  document: Attachments = dspy.InputField(desc="the current document to analyze")
  categories: list[str] = dspy.InputField(desc="the categories to generate a datapoint for based on what's in the pdf.")
  last_context: str = dspy.InputField(desc="Important context of the overall goal and previous steps from other agents.")
  next_context: str = dspy.OutputField(desc="The input context, where new information is optionally added on if thought to be important. New context, if present at all, should be brief to ensure the overall context doesn't get too long.")
  csv_row: str = dspy.OutputField(desc="a row of csv data that summarizes the document based on the given categories.")
  
  class doc_analyzer_nosql(dspy.Signature):
    """You will receive a single document from a collection, and using the given categories you will update the csv file to include the current documents's information."""
    document: Attachments = dspy.InputField(desc="the current document to analyze")
    categories: list[str] = dspy.InputField(desc="the categories to generate a datapoint for based on what's in the document.")
    in_csv: str = dspy.InputField(desc="The csv file being used to store information about the collection of documents.")
    last_context: str = dspy.InputField(desc="Important context of the overall goal and previous steps from other agents.")
    next_context: str = dspy.OutputField(desc="The input context, where new information is optionally added on if thought to be important. New context, if present at all, should be brief to ensure the overall context doesn't get too long.")
    out_csv: str = dspy.OutputField(desc="The csv file updated to include the current document in it's information.")

class collection_analyzer(dspy.Module):
  def __init__(self):
  super().__init__()
self.doc_analyzer_sql = dspy.ChainOfThought(doc_analyzer_sql)
self.doc_analyzer_nosql = dspy.ChainOfThought(doc_analyzer_nosql)

def forward(self, documents: list[Attachments]):
  for document in Attachments:
  