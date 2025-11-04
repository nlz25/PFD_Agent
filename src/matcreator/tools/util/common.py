import traceback
import time
import uuid
import os

def generate_work_path(create: bool = True) -> str:
    """
    Generate a unique working directory path based on call function and current time.
    
    directory = calling function name + current time + random string.
    
    Returns:
        str: The path to the working directory.
    """
    calling_function = traceback.extract_stack(limit=2)[-2].name
    current_time = time.strftime("%Y%m%d%H%M%S")
    random_string = str(uuid.uuid4())[:8]
    work_path = f"{current_time}.{calling_function}.{random_string}"
    if create:
        os.makedirs(work_path, exist_ok=True)
    
    return work_path