"""
code to test the programming side of category_analysis code. This does not test the command line interface, just calling the modules in code.
"""
#import necessary packages
from category_analysis import *
import os
import mlflow

#set up dspy
api_key = os.environ['paul2']
lm = dspy.LM('gemini/gemini-2.5-flash', api_key=api_key, max_tokens=8000)
dspy.configure(lm=lm)

#set up mlflow
mlflow_tracking_uri = "../../mlflow/experiments"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("CategoryFinder")
mlflow.dspy.autolog()

#set up documents and categories for input data. run documents through Attacments api to get Attachments objects.
categories = [
    "document", "number of violations", "list and details of violations"
]
documents = []
documents.append(
    Attachments(
        "TrainingData/32037 ALR10C6BZ 097 07-10-2025 INSPR AEPACS NA.pdf"))
documents.append(
    Attachments(
        "TrainingData/52781 ASN62022-CSW 097 07-10-2025 INSPR AEPACS Deer Crest.pdf"
    ))
documents.append(
    Attachments(
        "TrainingData/61763 ALR10C6PZ 077 07-10-2025 INSPR AEPACS NA.pdf"))

#run documents through trend_analyzer
analyze_categories = categories_analyzer()
result, context = analyze_categories(documents=documents,
                                 context="")

#print results
print(result)
print(context)