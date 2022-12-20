from pprint import pprint
from paddlenlp import Taskflow

if __name__ == '__main__':
    schema = ['出发地', '目的地', '费用', '时间']
    # 设定抽取目标和定制化模型权重路径
    my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
    print(my_ie("城市内交通费7月5日金额114广州至佛山"))