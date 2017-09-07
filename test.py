import os
rootdir="C:\\Users\\its\\Documents\\code\\stacking\\tensor\\"
link_dirs=os.listdir(rootdir)
for i in range(len(link_dirs)):
    link_path=os.path.join(rootdir,link_dirs[i])
    if(os.path.isdir(link_path)):
        link=os.path.basename(link_path)
        print(link)