from modules import *
import os
import mlflow

api_key = os.environ['paul2']
lm = dspy.LM('gemini/gemini-2.5-flash', api_key=api_key, max_tokens=8000)
dspy.configure(lm=lm)

mlflow_tracking_uri = "../../mlflow/experiments"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("TrendFinder")
mlflow.dspy.autolog()

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

analyze_trends = trend_analyzer()
result, context = analyze_trends(documents=documents,
                                 categories=categories,
                                 context="")
print(result)
print(context)
