from Thread.Worker.Helper import Helper
from Thread.Worker.Trainer import Trainer
from Thread.Worker.BaseModel import *
import numpy

class Client_info:

    def __init__(self, round_ID: int, host: str, port: int, DH_public_key: int):
        # Communication
        self.round_ID = round_ID
        self.host = host
        self.port = port


# class Commiter:

#     def __init__(self, params : tuple[int]):
#         self.p = params[0]
#         self.h = params[1]
#         self.k = params[2]
#         self.r = None

#     def commit(self, data) -> numpy.uint64:
#         assert self.r
#         data = int(data)
#         return (Helper.exponent_modulo(self.h, data, self.p) * Helper.exponent_modulo(self.k, self.r, self.p)) % self.p

#     def check_commit(self, data: numpy.ndarray[numpy.float32 | numpy.int64], commit: numpy.ndarray[numpy.uint64]) -> bool:
#         assert self.r
#         if len(data) != len(commit):
#             print(f"Model parameters length: {len(data)}, type {type(data[0])}")
#             print(f"Model commmit lenght: {len(commit)}, type {type(commit[0])}")
#             return False
#         return all(self.commit(data[idx]) == commit[idx] for idx in range(len(data)))

#     def set_secret(self, r: int):
#         self.r = r


class Aggregator_info:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    
class Manager:

    class FLAG:
        class NONE:
            # Default value
            pass
        class RE_REGISTER:
            # When commander wants to re-register
            pass
        class ABORT:
            # Used to send abort signal to Trusted party
            pass
        class STOP:
            # Used to stop processing
            pass
        class TRAIN:
            # Used to start training
            pass

    def __init__(self):
        # FL parameters
        # Communication
        self.host = "localhost"
        self.port = Helper.get_available_port()
        self.aggregator_info = None
        # Public parameters
        # Controller
        self.flag = self.FLAG.NONE
        self.abort_message = ""
            # Trainer
        self.trainer = None
            # Signer

        # Round parameters
        self.round_ID = None

    def get_flag(self) -> type:
        if self.flag == Manager.FLAG.NONE:
            return Manager.FLAG.NONE
        print(f"Get flag of {self.flag.__name__}")
        return_flag = self.flag
        self.flag = Manager.FLAG.NONE
        return return_flag

    def set_flag(self, flag: type) -> None:
        self.flag = flag
        print(f"Set flag to {self.flag.__name__}")

    def set_FL_public_params(self, aggregator_host: str, aggregator_port: int,  model_type: type):
        self.aggregator_info = Aggregator_info(aggregator_host, aggregator_port)
        self.trainer = Trainer(model_type)

    def set_round_information(self, round_number: int, round_ID: int):
        self.round_number = round_number
        self.round_ID = round_ID
    
    # def set_last_commit(self, commit: numpy.ndarray[numpy.uint64]):
    #     self.last_commit = commit

    def get_model(self):
        return self.trainer.get_parameters()

    def abort(self, message: str):
        self.abort_message = message
        self.set_flag(self.FLAG.ABORT)
    
    def start_train(self):
        self.trainer.set_dataset_ID(self.round_ID, self.round_number)
        self.trainer.train()

    def test_aggregated_model(self) -> float:
        """Test received aggregated model on client's test dataset and return accuracy"""
        return self.trainer.test()