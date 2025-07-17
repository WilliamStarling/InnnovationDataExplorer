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
    in_csv: str = dspy.InputField(
        desc=
        "The csv file being used to store information about the collection of documents. elements are seperated by commas, new rows by new line characters."
    )
    last_context: str = dspy.InputField(
        desc=
        "Important context of the overall goal and previous steps from other agents."
    )
    next_context: str = dspy.OutputField(
        desc=
        "The input context, where new information is optionally added on if thought to be important. New context, if present at all, should be brief to ensure the overall context doesn't get too long."
    )
    out_csv: str = dspy.OutputField(
        desc=
        "The csv file updated to include the current documents information. elements are seperated by commas, new rows by new line characters."
    )


class trend_analyzer(dspy.Module):

    def __init__(self):
        super().__init__()
        self.doc_analyzer_sql = dspy.ChainOfThought(doc_analyzer)

    def forward(self, documents: list[Attachments], categories: list[str],
                context: str):
        doc_summary = ""
        for document in documents:
            result = self.doc_analyzer_sql(document=document,
                                           categories=categories,
                                           in_csv=doc_summary,
                                           last_context=context)
            context = result.next_context
            doc_summary = result.out_csv
        print(doc_summary)
        return doc_summary, context
