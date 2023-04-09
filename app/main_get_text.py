from fastapi import FastAPI
from http import HTTPStatus
from enum import Enum
import re
from typing import TypedDict

class InputData(TypedDict):
    email: str
    domain_match: str

class ItemEnum(Enum):
   alexnet = "alexnet"
   resnet = "resnet"
   lenet = "lenet"

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
   return {"item_id": item_id}

@app.get("/query_items")
def read_item(item_id: int):
   return {"item_id": item_id}


database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
   username_db = database['username']
   password_db = database['password']
   if username not in username_db and password not in password_db:
      with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
      username_db.append(username)
      password_db.append(password)
   return "login saved"

@app.post("/text_model/")
def contains_email(data: InputData):
    # Extract the 'email' and 'domain_match' values from the input 'data' dictionary
    email = data.get("email")
    domain_match = data.get("domain_match")

    # Check if the 'email' value is missing and return a '400 Bad Request' response if so
    if email is None:
        return {
            "input": data,
            "message": "Missing 'email' field",
            "status-code": HTTPStatus.BAD_REQUEST
        }

    # Check if the 'domain_match' value is invalid and return a '400 Bad Request' response if so
    if domain_match not in ["gmail", "hotmail"]:
        return {
            "input": data,
            "message": f"Invalid 'domain_match' field: {domain_match}",
            "status-code": HTTPStatus.BAD_REQUEST
        }

    # Define a regex pattern for validating email addresses
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Check if the email format is invalid and return a '400 Bad Request' response if so
    match = re.fullmatch(regex, email)
    if match is None:
        return {
            "input": data,
            "message": "Invalid email format",
            "status-code": HTTPStatus.BAD_REQUEST
        }

    # Extract the domain from the email address
    domain = email.split("@")[1]

    # Create a response dictionary with the input 'data', a '200 OK' status, and two boolean values
    # indicating whether the email matches the specified format and whether it matches the specified domain
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": True if match is not None else False,
        "matches_domain": domain == domain_match
    }

    # Return the response dictionary as the HTTP response
    return response
