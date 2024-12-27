import os

def convert_files_to_lowercase(folder_path):
    # Walk through all files and directories in the given folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Process only files with the .F90 extension
            if file.endswith(".o.d"):
                file_path = os.path.join(root, file)
                
                # Open the file, read its content, and convert to lowercase
                with open(file_path, 'r') as f:
                    content=""
                    line = f.readline().lower()
                    while line != "":
                        
                        if line.startswith("#"):
                            content = content + "! "+line
                        else:
                            content = content + line    
                        line = f.readline().lower()    
                
                # Write the modified content back to the file
                with open(file_path, 'w') as f:
                    f.write(content)
                
                print(f"Converted {file_path} to lowercase.")



def switch_files(folder_path):
    # Walk through all files and directories in the given folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Process only files with the .F90 extension
            if file.endswith(".o.d"):
                file_path = os.path.join(root, file)
                new_file_path = file_path.replace(".o.d", ".F90")
                
                # Open the file, read its content, and convert to lowercase
                try:
                    with open(file_path, 'r') as f:
                        content=f.read()
                    with open(new_file_path, 'r') as f2:
                        content2=f2.read()
                        
                    # Write the modified content back to the file
                    with open(new_file_path, 'w') as f:
                        f.write(content)
                    with open(file_path, 'w') as f2:
                        f2.write(content2)
                except:
                    print("Error switching files" + file_path)
                
                print(f"Switched {file_path} around.")

# Specify the folder path to start the recursive conversion
folder_path = "/home/alex/icon-model/"
convert_files_to_lowercase(folder_path)
switch_files(folder_path)