from typing import Iterable, List, Dict, Tuple

DEFAULT_VALUE = '<_UNK>'


class Alphabet(object):
    def __init__(self, iterable: Iterable[str], offset: int) -> None:

        self.instances: List[str] = list(iterable)
        self.instance2index: Dict[str, int] = {k: i + offset for i, k in enumerate(self.instances)}
        self.offset: int = offset

    def get_index(self, instance: str) -> int:
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.offset != 0:
                return 0
            else:
                raise KeyError("instance not found: {:s}".format(instance))

    def get_instance(self, index: int) -> str:
        if self.offset != 0 and index == 0:
            # First index is occupied by the wildcard element.
            return DEFAULT_VALUE
        else:
            try:
                return self.instances[index - self.offset]
            except IndexError:
                raise IndexError("unknown index: {:d}".format(index))

    def size(self) -> int:
        return len(self.instances) + self.offset


# misc_config is dic that is generated according to dataset
def load_dynamic_config(misc_dict: Dict[str, Alphabet]) -> Tuple[Alphabet, Alphabet]:
    voc_dict = misc_dict['voc_dict']
    label_dict = misc_dict['label_dict']

    return voc_dict, label_dict


def save_dynamic_config(reader) -> Dict[str, Alphabet]:
    misc_dict = dict()
    misc_dict['voc_dict'] = reader.subword_alphabet
    misc_dict['label_dict'] = reader.label_alphabet

    return misc_dict
