# Data Analysis
Handled by William Starling (williamjonas@comcast.net) and Byron Overton.

The purpose of this stage is to take the doucments found with Data Collection, and analyze them for information relevent to what the user is doing. It then will compile this into a csv file to be used by the Data formatting stage. It's able to do all of this using DSPy.

## Project Folder Structure
- AICategorizerwiRera: Directory (Byron's categorization code, limited functionality)
  - categroizer.py (file, the main categroization code. run this file using python categorizer.py in order to demo it.)
- results: Directory (output results from running through categorization and analysis pipeline)
- TrendFinder: Directory (William's document analysis code. Also houses William replacement categorization code.)
  - optimized_modules: Directory (meant to hold the DSPy modules after they had been optimized. Was unable to finish code to properly optimize modules however.)
  - results: Directory (output results from William's categorization code after being ran in demo mode.)
  - TrainingData: Directory (example pdfs to be used to test both categorization code and analysis code)
  - category_analysis.py: Python File (William's DSPy AI categorization code. run with "python category_analysis.py" to demo it.)
  - category_test.py: python file (a file I wrote for testing my categorization code. run it if you wish, and read to see how to call stuff, but otherwise not needed.)
  - trend_analysis.py: Python File (William's document analysis code. run it using "python trend_analysis.py" in order to demo it.)
  - trend_test.py: python file (a file I wrote for testing my document analysis code. run it if you wish, and read to see how to call stuff, but otherwise not needed.)
- analysis_pipeline.py: python file (code I had generated in order to enable use of Byron's categorization code and my document analysis code as one. Byron's code would need to be re-worked or finished and this file would need to be rewritten as a result to be of any use.)
- analysis_test.py: python file (This was a test file I wrote in order to test calling analysis_pipeline.py)
- analysis_will.py: python file (this is a pipeline file I wrote in order to enable using my (William Starling's) DSPy powered categorization code and Document Analysis code together as one. run it using "python analysis_will.py" in order to demo.)
## How It Works
There are two substages to the data Analysis stage. First, the documents are analyzed with the goal of finding categories/fields of data that are relevent to what the user is trying to do. Second, it re-analyzes the documents in order to find information that fits into these categories. it places these categories and data points into a csv file.

There is something important worth noting. I (William Starling) handled collecting data from the documents, and Byron handled finding the categories of information. However, Byron's method used Regualar Expressions to parse the documents for categories, which did not work well in actual use cases given how generalized we want it to be able to work. The AI categorization was also not working properly. As a result, I (William Starling) wrote my own quick categorization DSPY code using the same DSPy structure that I used for gathering data. This can be used until certain parts of byron's code is fixed. that is why there is 2 types of categorization in this directory. one by me and one by Byron. I am stating this now to avoid confusion in the future.

Analyzing the documents for data happens using a step-by-step iterative process. It uses a dspy Signature that takes a document, a list of categories, context from the initial instructions and notes from the agents from previous stages, and a csv file. For each document, this siganture is used until they all have been gone through. The context and CSV files are meant to be passed back into the siganture for the next document, iteratively updated with each document.
![alt text](https://github.com/WilliamStarling/InnnovativeDataLanguageExplorer/blob/william_categorizer/IDLE.jpg?raw=true)

My categorization code works very similarly, except it doesn't take a csv file for input, and it has a list of categories as the ouput. This list of categories is outputted each time to act as input for the next document, and is iteratively updated (along with the context).