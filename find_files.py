import os

def find_ignore():
    #ig = os.path.join(root,"/.ignore")
    #ig = root + "/.ignore"
    #print(ig)
    #os.path.isfile(ig)
    return(True)

def get_file(file_type,root,file,file_vec):
    if(file.split(".")[-1] == file_type):
        f = os.path.join(root,file)
        file_vec.append(f)

supported_files = ['snp','msh']
def find_file(file_type,file_vec):
    for root, dirs, files in os.walk(".", topdown=True):
    
        if (find_ignore()): 
            for file in files:
                if(file_type in supported_files):
                    get_file(file_type,root,file,file_vec)
                else:
                    print("find_files: Wrong type of extension")

    return(file_vec)

