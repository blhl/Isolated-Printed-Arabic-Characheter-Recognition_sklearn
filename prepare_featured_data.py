import pandas as pd
import numpy as np
import os
#os.chdir("/mydir")
#for file in glob.glob("*.txt"):
#    print(file)

#featuretype = "projX_H32"
featuretype = "projX_H64"
#featuretype = "projX_H64FFT"

settype = "Test"
imgfolders="Data_img/" + settype + "/"

subfolders = [f.path for f in os.scandir(imgfolders) if f.is_dir()]
#print('subfolders', subfolders)
i=0
for subfolder in subfolders:
    img_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
    #print('img_files', img_files)
    # imgfile="1_TIN102_1.png"
    for imgfile in img_files:
        str_ext = imgfile.split(".")
        str_chr = str_ext[0].split("_")
        chr_id=str_chr[0]
        chr_code=str_chr[1]
        chr_seq=str_chr[2]

        #print(str_ext, chr_id, chr_code, chr_seq)
        featurefile=chr_code+"."+chr_seq
        df = pd.read_fwf("Features/"+featuretype+"/"+featurefile)
        #data.append(int(chr_id))
        df.loc[-1] = [int(chr_id)]  # adding a row
        #data.index = data.index + 1  # shifting index
        #data = data.sort_index()  # sorting by index

        #ls=[]
        #ls.append([data.T, int(chr_id)])
        #print(df.T)
        if i==0:
            df_all=df.T.copy()
        else:
            df_all=pd.concat([df_all, df.T], axis=0)

        #df_all.append(range(1,33))
        i=i+1


print("df_all", df_all.shape)
df_all.to_csv("Data_csv/" + settype + "_"+ featuretype +".csv", header=False, index=False)