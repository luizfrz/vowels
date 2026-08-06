import kagglehub
import os
import  shutil
data_dir = '../data'

cache_path  = kagglehub.dataset_download("crawford/emnist")

if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    print("Pasta criada com sucesso.", os.listdir(data_dir) )
else:
    print("Pasta ja existe:", os.listdir(data_dir)) 

for filename in os.listdir(cache_path):
    origin = os.path.join(cache_path, filename)
    destin = os.path.join(data_dir, filename)
    
    if os.path.isdir(origin):
        shutil.copytree(origin, destin, dirs_exist_ok=True)
    else:
        shutil.copy(origin, destin)
        
print("Dataset baixado com sucesso", os.path.abspath(data_dir))