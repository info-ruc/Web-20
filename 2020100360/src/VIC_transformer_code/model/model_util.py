#!usr/bin/env python
# coding:utf-8
import codecs as cs
import torch


def init_tensor(tensor, low=0, high=1,mean=0, std=1, activation_type="linear"):
    """Init torch.Tensor
    Args:
        tensor: Tensor to be initialized.
        low: The lower bound of the uniform distribution,
            useful when init_type is uniform.
        high: The upper bound of the uniform distribution,
            useful when init_type is uniform.
        mean: The mean of the normal distribution,
            useful when init_type is normal.
        std: The standard deviation of the normal distribution,
            useful when init_type is normal.
        activation_type: For xavier and kaiming init,
            coefficient is calculate according the activation_type.
    Returns:
    """
    return torch.nn.init.uniform_(tensor, a=low, b=high)


def get_optimizer(config, params):
    params = params.get_parameter_optimizer_dict()
    return torch.optim.Adam(lr=config.optimizer.learning_rate,
                                params=params)


def get_hierar_relations(hierar_taxonomy, label_map):
    """ get parent-children relationships from given hierar_taxonomy
        hierar_taxonomy: parent_label \t child_label_0 \t child_label_1 \n
    """
    hierar_relations = {}
    with cs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.strip("\n").split("\t")
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                continue
            parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
    return hierar_relations
