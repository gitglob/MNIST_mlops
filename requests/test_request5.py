import requests
pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

print(response.status_code)

if response.status_code == 200:
   print('Success!')
elif response.status_code == 404:
   print('Not Found.')

print(f"Content: {response.content}")
print(f"Type: {type(response.content)}")

