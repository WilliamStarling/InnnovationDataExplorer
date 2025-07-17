#For setting up and creating the DSPy signatures and modules needed to search for trends as a part of the bigger data analysis agent.
import dspy
from attachments.dspy import Attachments
import pandas as pd

# Print the dspy version being used
print(f"DSPy version: {dspy.__version__}")

class doc_analyzer(dspy.Signature):
  """You will receive a single document from a collection, and using the given categories you will update the csv file to include the current documents's information."""
  document: Attachments = dspy.InputField(
      desc="the current document to analyze")
  categories: list[str] = dspy.InputField(
      desc=
      "the categories to generate a datapoint for based on what's in the document."
  )
  in_df: pd.DataFrame = dspy.InputField(
      desc=
      "The pandas dataframe being used to store information about the collection of documents."
  )
  last_context: str = dspy.InputField(
      desc=
      "Important context of the overall goal and previous steps from other agents."
  )
  next_context: str = dspy.OutputField(
      desc=
      "The input context, where new information is optionally added on if thought to be important. New context, if present at all, should be brief to ensure the overall context doesn't get too long."
  )
  out_df: str = dspy.OutputField(
      desc=
      "The pandas dataframe updated to include the current documents information."
  )

"""
class trend_analyzer(dspy.Module):
  def __init__(self):
    super().__init__()
    self.doc_analyzer_sql = dspy.ChainOfThought(doc_analyzer)
    
  def forward(self, documents: list[Attachments], categories: list[str], context: str):
    doc_summary = pd.DataFrame(columns=categories)
    for document in documents:
      result = self.doc_analyzer_sql(document=document, categories=categories, in_df = doc_summary, last_context=context)
      context = result.next_context
      doc_summary = result.out_csv
      documents_csv = doc_summary.to_csv(index=False)
    return document_csv, context
"""