import os
import shutil
def cp_img(test_img,result_img,cpto):
    # cpto = './result/'
# path1 = './train1'
# path2 = './test'
    test_img_file = os.path.split(test_img)[-1]
    result_img_file = os.path.split(result_img)[-1]

    test_img_name = test_img_file.split('.')[0]
    cpto_dir = os.path.join(cpto,test_img_name)
    if not os.path.isdir(cpto_dir):
        os.makedirs(cpto_dir)
    test_img_todir = os.path.join(cpto_dir,test_img_file)
    result_img_todir =  os.path.join(cpto_dir,result_img_file)
    shutil.copyfile(test_img,test_img_todir)
    shutil.copyfile(result_img,result_img_todir)        
    

   
