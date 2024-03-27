import random
def wildcard_scene_def(directory="./wildcards",file_name="example"):
    if file_name != "":
        if directory[-1] != "/":
            directory = directory + "/"
        directory_path = f"{directory}{file_name}.txt"
        try:
            with open(directory_path, "r") as file:
                lines = file.readlines()
                if lines:
                    random_line = random.choice(lines)
                    return random_line.strip()
                else:
                    print("The file is empty.")
        except FileNotFoundError:
            print(f"The file {file_name}.txt was not found.")
    else:
        print("The file name cannot be empty.")
    return ""