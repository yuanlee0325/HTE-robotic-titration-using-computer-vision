from typing import NamedTuple, List, Callable, List, Tuple, Optional, Union

class DataStruct(NamedTuple):
    lab : Optional[List] = None # input dimension
    rgb : Optional[List] = None # hidden layers including the output layer
    laberr : Optional[List] = None
    rgberr : Optional[List] = None
    bbox : Optional[List] =None
    mask : Optional[List] = None
    t : Optional[List] = None
    #activations : List[Optional[Callable[[torch.Tensor],torch.Tensor]]] # list of activations
    #bns : List[bool] # list of bools
    #dropouts : List[Optional[float]] # list of dropouts probas