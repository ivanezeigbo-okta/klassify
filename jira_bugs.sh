ALL_BUGS=$(curl \
   -u $myEmailAddress:$myApiToken \
   -X GET \
   -H "Content-Type: application/json" \
   -H "Cache-Control: no-cache" \
   'https://oktainc.atlassian.net/rest/api/2/search?jql=project=OKTA%20AND%20type=bug&maxResults=100&startAt=0&fields=description,issuetype,status,summary,assignee,components,issuelinks')
ls > all_bugs.json
echo $ALL_BUGS > all_bugs.json
#jq '.issues | length' Documents/all_bugs.json
