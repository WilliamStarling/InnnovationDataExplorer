import dspy 
import os 
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM("gemini/gemini-2.0-flash-lite", api_key=os.getenv('GOOGLE_API_KEY'))
dspy.configure(lm=lm)

class ExtractFields(dspy.Signature):
    """
    Given a sentence, extract the fields if they are present, if they are not present return None. if the number to download is unspecified, then download 10
    
    media_area options: Air, Land, or Water
    county options: Autauga, Baldwin, Barbour, Bibb, Blount, Bullock, Butler, Calhoun, Chambers, Cherokee, Chilton, Choctaw, Clarke, Clay, Cleburne, Coffee, Colbert, Conecuh, Coosa, Covington, Crenshaw, Cullman, Dale, Dallas, DeKalb, Elmore, Escambia, Etowah, Fayette, Franklin, Geneva, Greene, Hale, Henry, Houston, Jackson, Jefferson, Lamar, Lauderdale, Lawrence, Lee, Limestone, Lowndes, Macon, Madison, Marengo, Marion, Marshall, Mobile, Monroe, Montgomery, Morgan, Perry, Pickens, Pike, Randolph, Russell, St. Clair, Shelby, Sumter, Talladega, Tallapoosa, Tuscaloosa, Walker, Washington, Wilcox, Winston.
    document_category options: Complaints, Education & Outreach, Enforcement, General Correspondence, Inspections, Monitoring, Other, Permitting, Public Notices
    """
    sentence: str = dspy.InputField()

    media_area: str = dspy.OutputField()
    facility: str = dspy.OutputField()
    permit_number: str = dspy.OutputField()
    county: str = dspy.OutputField()
    #file_name 
    srf_number: str = dspy.OutputField()
    #document_date: str = dspy.OutputField()
    document_category: str = dspy.OutputField()
    number_of_files_to_download: str = dspy.OutputField()

def DspyFieldExtractor(sentence):
    extract = dspy.ChainOfThought(ExtractFields)
    output = extract(sentence=sentence)
    return output
