from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to a specific origin if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
class User(BaseModel):
    email: str
    password: str

@app.post("/signup")
async def signup(user: User):
    # Print email and password to the console
    print(f"Signup - Email: {user.email}, Password: {user.password}")
    
    # Return a response
    return {"message": "Signup successful"}

@app.post("/login")
async def login(user: User):
    # Print email and password to the console
    print(f"Login - Email: {user.email}, Password: {user.password}")
    
    # Return a response
    return {"message": "Login successful"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
