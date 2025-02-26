#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('Wente47/M2E', subset_name='default', split='train')