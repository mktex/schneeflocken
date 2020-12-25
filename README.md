<br />
<p align="center">
  <h3 align="center">Analysis and modeling of wages for data science jobs
    according to poll results from StackOverflow 2019, 2020</h3>
</p>

### About

The jupyter notebook was used to analyse data from StackOverflow polls
from 2019 and 2020 with the aim at answering questions related to
average income and job satisfaction of data scientists

- Gather and assess data:
  - Startup (for running need here local paths)
  - Data loading, target and id variables
  - Two possible target variables: Job Satisfaction and Income

- Cleaning and visualisations:
  - Replacements and filtering
  - Handling missing data
  - Exploratory Analysis

- Model:
  - Handling multicolinearity and dimensionality reduction
  - Linear Model


### Built With

Following Python libraries are required
* pandas
* numpy
* scipy
* statsmodels
* seaborn
* matplotlib
* scikit-learn

For visualisation during the EA phase Rapidminer was used due its
graphical capabilities.

For visualizations that were published in the blog article, Datawrapper
has been used.

<!-- GETTING STARTED -->
### Getting Started

Download the repository and start the jupyter session

   ```sh
   git clone https://github.com/mktex/schneeflocken.git
   cd ./schneeflocken/notebooks
   jupyter notebook
   ```

  In order for the notebook to run properly one needs to set the
  variable **project_dir** at start (3rd paragraph)  
  to the path of local ./schneeflocken/notebooks folder.  
  This is to make possible loading the three modules under /eda.

  Additionaly the poll data (survey_results_public.csv and
  survey_results_schema.csv) needs to be unzipped under
  ./notebooks/data/2020 and ./notebooks/data/2019 respectively

Beside matplotlib in Python, Rapidminer and Datawrapper were used to
build several visualisations.

Blog post: http://machine-learning.bfht.eu/so-what-would-be-a-good-fair-price-for-a-data-scientist
