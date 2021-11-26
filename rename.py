import glob 
import os 

root = './a/'
all_files = os.listdir(root)

print(all_files)

for file in all_files:
    print(file)
    file_rename = file.split('.')[0].replace(" ", "") + '.' +file.split('.')[-1]
    os.rename(root + file, root +file_rename )

