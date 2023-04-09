import requests
response = requests.get('https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*')

print(response.status_code)

if response.status_code == 200:
   print('Success!')
elif response.status_code == 404:
   print('Not Found.')

print(f"Content: {response.content}")
print(f"Type: {type(response.content)}")

with open(r'requests/img.png','wb') as f:
   f.write(response.content)
