import os


def deleteIgnoreFile(file_list):
    '''
    移除隐文件
    '''
    for item in file_list:
        if item.startswith('.'):# os.path.isfile(os.path.join(Dogs_dir, item)):
            file_list.remove(item)
    return file_list

def getClasses(dir_path):
    '''
    得到分类目录的分类列表
    '''
    classes_name_list = os.listdir(dir_path)
    classes_name_list = deleteIgnoreFile(classes_name_list)
    classes_name_list.sort()  # 字典序
    return classes_name_list