"""code I created to test the analysis_pipeline.py code."""
from TrendFinder.trend_analysis import *
import analysis_pipeline
import os
import asyncio
from AICategorizerwiReRa.categorizer import *

api_key = os.environ['paul2']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
lm = dspy.LM('gemini/gemini-2.5-flash', api_key=api_key, max_tokens=8000)
dspy.configure(lm=lm)

#asyncio.run(analysis_pipeline.example_usage())

manager = analysis_pipeline.PipelineManager()

documents = []
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

# Run with auto-discovered categories
result = asyncio.run(
    manager.run_pipeline(
        documents=documents,
        initial_context=
        "I am trying to find out about what issues construction companies are causing. Please anayze these documents to find out how many violations these companies are causing, and if there's a pattern between them.",
        lm_model=lm))

# Display and save results
manager.display_results(result)
manager.save_results(result)