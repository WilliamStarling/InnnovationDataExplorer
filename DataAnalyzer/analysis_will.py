"""
This code is called analysis_will. because I'm only using the things  I made. Most notably, I'm using my categorizer code instead of Byrons. Oh yeah by "I" I mean William Starling (williamjonas@comcast.net, 251-680-9048).
The purpose of this code is to bring together the categorizer code and trend analyzer code to allow for a seamless analysis of documents, with only a context string and a list of documents needed as inputs.
"""

#import needed packages
from TrendFinder.trend_analysis import trend_analyzer
from TrendFinder.category_analysis import categories_analyzer
import dspy
import os
import mlflow
from attachments.dspy import Attachments

"""module first analizes documents for categories, then uses those categories to analyze the documents for the information in them. use forward function to run.
inputs: documents: list of Attachments (a list of documents to be analyzed, as attachment objects. see trend_test.py for an example of how it's used.)
        context: string (the context of the overall goal and previous steps from other agents.)
outputs: analysis: string (the final analysis of the documents, in csv format.)
        context: string (the updated context of the instructions, previous agents notes, and this agents notes.)"""
class document_analysis(dspy.Module):
  def __init__(self):
    super().__init__()
    self.categories_analyzer = categories_analyzer()
    self.trend_analyzer = trend_analyzer()
    self.context = ""
    self.documents = []
    self.categories = []
    self.analysis = ""

  def forward(self, documents: list[Attachments], context: str):
    self.documents = documents
    self.context = context

    print("Beginning search for categories...")
    self.categories, self.context = self.categories_analyzer.forward(documents=self.documents, context=self.context)

    print("Categories found. Beginning trend analysis...")
    self.analysis, self.context = self.trend_analyzer.forward(documents=self.documents, categories=self.categories, context=self.context)
    print("Analysis complete.")
    return self.categories, self.context
  
def example_usage(documents, context, analysis_result, analyze_documents):
  documents.append(
    Attachments(
        "TrendFinder/TrainingData/32037 ALR10C6BZ 097 07-10-2025 INSPR AEPACS NA.pdf"
    ))
  documents.append(
    Attachments(
        "TrendFinder/TrainingData/52781 ASN62022-CSW 097 07-10-2025 INSPR AEPACS Deer Crest.pdf"
    ))
  documents.append(
    Attachments(
        "TrendFinder/TrainingData/61763 ALR10C6PZ 077 07-10-2025 INSPR AEPACS NA.pdf"
    ))
  context = "I am trying to find out about what issues construction companies are causing. Please anayze these documents to find out how many violations these companies are causing, and if there's a pattern between them, and if there's any enviromental effects."

  #run documents through trend_analyzer
  analysis_result, context = analyze_documents.forward(documents=documents, context=context)
  return analysis_result, context

def print_results(analysis_result, context):
  print("Analysis complete. Here are the results:")
  print("=" * 60)
  print("Context:")
  print(context)
  print("=" * 60)
  print("Analysis:")
  print(analysis_result)

if __name__ == "__main__":
    #set up dspy
    api_key = os.environ['paul2']
    lm = dspy.LM('gemini/gemini-2.5-flash', api_key=api_key, max_tokens=8000)
    dspy.configure(lm=lm)

    #setup mlflow
    mlflow_tracking_uri = "../../mlflow/experiments"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("CategoryTrendFinder")

    #setup variables
    analyze_documents = document_analysis()
    documents = []
    context = ""
    analysis_result = ""
  
    #check if user wants to use the default testing data or not.
    answer = input("use custom inputs? (y/n): ")
    if answer.lower() == "n":
      analysis_result, context = example_usage(documents, context, analysis_result, analyze_documents)

    #print the results!
    print_results(analysis_result, context)
      
        