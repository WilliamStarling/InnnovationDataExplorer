# Innnovative Data Language Explorer (I.D.L.E)
A project created for the Innovation Portal as a part of our internship. Given data, or a website with data, the agent will gather the data, analyze it, and output it in the format specified with the information specified.

The idea behind it is there is three stages. 1. Data Collection (handled by Kingston Barnes, inside DataCollector Directory), 2. Data Analysis (handled by William Starling and Byron Overton, inside DataAnalyzer directory), and 3. Data Formatting (Handled by Amelia Moore and Joshua Brown, inside DataFortmatter directory). read the README file inside each of their documents for more insight.

This project was not finished by the deadline. So each step of the process will need to be ran individually, and unforunately they are not integrated together. Navigate to each director to learn about how to run it. If that is not specified, you may have to contact the person responsible to learn how to properly utilize it. What was finished, I (William Starling) will lay out here to the best of my extent. I only worked on the data analysis though, and for the other steps you will have to talk to the respective interns to understand the structure and how to use them more. I am more then happy to answer any questions you might have, or help you connect with the other interns if Paul hasn't already. I can be reached at williamjonas@comcast.net

## Project Folder Structure
- DataAnalyzer: Directory (Handled Data Analysis, stage 2 in IDLE)
- DataCollector: Directory (Handled collecting documents and other relevent data, stage 1 in IDLE)
- DataFormatter: Directory (Handles formatting the data for user presentation)
- mlflow: Directory (stores the tracing calls logged with mlflow)
- IDLE.jpg (diagram of program software, details on Analysis stage.)

## Dependencies
* attachments 0.21.0
* dspy-ai 2.6.27
* jupyter 1.1.1
* mlflow 3.2.0
* pandas 2.3.1

## Viewing MLflow information
Run the command "mlflow ui --backend-store-uri mlflow/experiments --host 0.0.0.0 --port 5000" in a terminal to startup the UI to access LM logs.
Choose your exeriment, then navigate to the Traces tab, and from there you can access each time the langauge model was ran through for each run of the program. This will tell you information about each input and output at each step of the process.

## Accessing Jupyter Notebooks
Access jupyter notebooks by running the command "jupyter notebook --ip=0.0.0.0 --port=5000 --allow-root --no-browser" in a terminal. If using Replit, you'll have to click .replit.dev in the address bar and click the dev url to open it in a new window to properly view it. Jupyter notebook files will end with the .ipynb file extension.