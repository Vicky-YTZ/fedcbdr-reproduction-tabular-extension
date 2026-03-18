import copy
import torch


def fedavg(local_models):
    """
    local_models: list of trained client models
    return: averaged global model
    """
    global_model = copy.deepcopy(local_models[0])

    global_state_dict = global_model.state_dict()

    for key in global_state_dict.keys():
        # 先取第一个模型参数
        global_state_dict[key] = local_models[0].state_dict()[key].clone()

        # 把其他模型对应参数加起来
        for i in range(1, len(local_models)):
            global_state_dict[key] += local_models[i].state_dict()[key]

        # 求平均
        global_state_dict[key] = global_state_dict[key] / len(local_models)

    global_model.load_state_dict(global_state_dict)

    return global_model
