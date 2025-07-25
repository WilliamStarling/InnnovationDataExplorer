import dspy 
import os 
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM("gemini/gemini-2.0-flash-lite", api_key=os.getenv('GOOGLE_API_KEY'))
dspy.configure(lm=lm)

class ExtractFields(dspy.Signature):
    """Given a sentence, extract the fields if they are present"""
    sentence: str = dspy.InputField()

    media_area: str = dspy.OutputField()
    facility: str = dspy.OutputField()
    permit_number: str = dspy.OutputField()
    county: str = dspy.OutputField()
    #file_name 
    srf_number: str = dspy.OutputField()
    document_date: str = dspy.OutputField()
    document_category: str = dspy.OutputField()

extract = dspy.ChainOfThought(ExtractFields)
sentence = "inspections in mobile county in the water"
output = extract(sentence=sentence)

for t in output:
    print(t)

print(output)