import requests
response = requests.get('https://api.github.com')

print(response.status_code)

if response.status_code == 200:
   print('Success!')
elif response.status_code == 404:
   print('Not Found.')