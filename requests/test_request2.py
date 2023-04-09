import requests
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")

print(response.status_code)

if response.status_code == 200:
   print('Success!')
elif response.status_code == 404:
   print('Not Found.')

print(f"Content: {response.content} \nType: {type(response.content)} \nFor humans: {response.json()}")