# klassify
Hackathon Project - Jira Bug Classification and Ranking

Klassify is a program that polls bugs from our Jira through an API, groups and classifies these bugs based on some similarity.
At core, it applies Machine Learning and Natural Language Processing algorithms to identify commonalities among Jiras. It uses the k means, a text clustering algorithm to group these Jiras. Ranks these classes based on number of related bugs.

## Advantages of Klassify

1. Identify Undetected Testing Gaps

2. Proper and Thorough Test Coverage

3. Detecting Commonly Reported Major Bugs

4. Prioritizing Testing Plans And Sprint Tasks

5. Identify Needed Functional-Related And Feature-Related Tests


## Getting Started

Before anything, make sure you have Python installed. This program is developed with Python.

1. Clone Git Repository

`git clone https://github.com/ivanezeigbo-okta/klassify.git`

2. Enter Directory And Create Virtual Environment

`python3 -m venv env`

3. Activate Virtual Environment

`source env/bin/activate`

4. Download dependencies

`pip install -r requirements.txt`

5. Replace `myEmailAddress` and `myApiToken` in `jira_bugs.sh`

You may also change the URL since this URL is adapted to only look at bugs submitted under the project, OKTA. You may change or delete the project parameter.

6. Run `jira_bugs.sh`

`bash jira_bugs.sh`

7. Finally run `run.py`

`python3 run.py`
