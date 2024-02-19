import random
def wildcard_scene_def(file_name, word, random_scene):
    if file_name != "":
        directory_path = f"./wildcards/{file_name}.txt"
        try:
            with open(directory_path, "r") as file:
                if random_scene:
                    lines = file.readlines()
                    if lines:
                        random_line = random.choice(lines)
                        return random_line.strip()
                    else:
                        print("The file is empty.")
                else: # not random_scene
                    """
                    for line in file:
                        if line.lower().startswith(word.lower()):
                            if ":" in line:
                                return line.split(":", 1)[-1].strip()  # Print everything after the first ':'
                            else:
                                return line.strip()
                    else:
                    """
                    print(f"No lines starting with '{word}' found in the file.")
        except FileNotFoundError:
            print(f"The file {file_name}.txt was not found.")
    else:
        print("The file name cannot be empty.")
    return ""

if __name__ == "__main__":
    wildcard_scene_def("", "", False)
