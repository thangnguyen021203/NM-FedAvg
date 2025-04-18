from Thread.Worker.Helper import Helper
from Thread.Worker.BaseModel import *           # This can be removed
import random, numpy
from sympy import randprime, primitive_root
from copy import deepcopy

class RSA_public_key:
    
    def __init__(self, e, n):
        self.e = e
        self.n = n

class Client_info:
        
    def __init__(self, ID: int, host: str, port: int):
        # Unique attributes
        self.ID = ID
        self.host = host
        self.port = port
        self.choose_possibility = 100
        # Round attributes
        self.round_ID = 0

    def set_round_information(self, client_round_ID: int):
        self.round_ID = client_round_ID

class Aggregator_info:

    def __init__(self, host: str, port: int, base_model_class: type):
        self.host = host
        self.port = port
        self.base_model_class = base_model_class

# class Commiter:

#     def __init__(self):
#         self.p = randprime(1 << 63, 1 << 64)
#         self.h = primitive_root(self.p)
#         self.k = random.randint(1 << 63, 1 << 64)
#         self.r = None


class Manager():

    class FLAG:
        class NONE:
            # Default value
            pass
        class START_ROUND:
            # When get initiation signal user
            pass
        class STOP:
            # Used to indicate situation that needs process stopping
            pass
        class TRAINING_COMPLETE:
            # Used to indicate training is complete
            pass

    def __init__(self):
        # FL parameters
            # Communication
        self.client_list : list[Client_info] = list()
        self.aggregator_info = None
            # Public parameters
        # self.commiter = Commiter()
        self.current_round = 0
        # self.last_commitment: numpy.ndarray[numpy.int64] = None
            # Controller
        self.flag = self.FLAG.NONE
        self.stop_message = ""
        # Round parameters
        self.round_manager : Round_Manager = None
        # Model accuracy tracking
        self.client_accuracies = {}  # Dict to store client_round_ID -> accuracy
        self.accuracy_threshold = Helper.get_env_variable("ACCURACY_THRESHOLD") 
        self.completion_threshold = Helper.get_env_variable("CLIENT_PERCENT_THRESHOLD") 

    def stop(self, message: str):
        self.stop_message = message
        self.set_flag(self.FLAG.STOP)

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

    def clear_client(self) -> None:
        self.client_list.clear()
    
    def clear_aggregator(self) -> None:
        self.aggregator_info = None
        # self.last_commitment = None

    def register_aggregator(self, host: str, port: int, base_model_class: type):
        self.aggregator_info = Aggregator_info(host, port, base_model_class)

    # def set_last_model_commitment(self, model_commitment: numpy.ndarray[numpy.int64]):
    #     self.last_commitment = model_commitment

    def __get_client_by_ID__(self, client_ID: int) -> Client_info | None:
        for client_info in self.client_list:
            if client_info.ID == client_ID:
                return client_info
        return None
    
    def __get_client_by_round_ID__(self, client_round_ID: int) -> Client_info | None:
        for client_info in self.client_list:
            if client_info.round_ID == client_round_ID:
                return client_info
        return None

    def add_client(self, client_id: int, host: str, port: int) -> None:
        self.client_list.append(Client_info(client_id, host, port))

    def get_current_round(self) -> int:
        return self.current_round

    # def get_commiter(self) -> Commiter:
    #     return self.commiter
    
    def choose_clients(self, client_num: int) -> list[Client_info]:
        if client_num > len(self.client_list):
            client_num = len(self.client_list)
        return_list = list()
        client_list = deepcopy(self.client_list)
        for i in range(client_num):
            chosen_one = random.choices(client_list, weights=[max(client.choose_possibility, 0) for client in client_list])[0]
            client_list.remove(chosen_one)
            return_list.append(chosen_one)
        return return_list
    
    def record_client_accuracy(self, client_round_id: int, accuracy: float) -> None:
        """Record a client's accuracy for the current round"""
        self.client_accuracies[client_round_id] = accuracy
        print(f"Received accuracy from client {client_round_id}: {accuracy:.2f}%")
        
        # Check if all clients have reported their accuracy
        if len(self.client_accuracies) == len(self.round_manager.client_list):
            self.evaluate_model_performance()
    
    def evaluate_model_performance(self) -> None:
        """Evaluate if training should complete or continue to next round"""
        if not self.client_accuracies:
            print("No accuracy data available for evaluation")
            return
            
        # Count clients meeting the accuracy threshold
        clients_meeting_threshold = sum(1 for acc in self.client_accuracies.values() if acc >= self.accuracy_threshold)
        total_clients = len(self.client_accuracies)
        percentage_meeting_threshold = clients_meeting_threshold / total_clients
        
        print(f"\n----- Model Performance Evaluation -----")
        print(f"Round {self.current_round} - Clients meeting {self.accuracy_threshold}% threshold: "
              f"{clients_meeting_threshold}/{total_clients} ({percentage_meeting_threshold*100:.1f}%)")
        
        if percentage_meeting_threshold >= self.completion_threshold:
            print(f"Training complete! {percentage_meeting_threshold*100:.1f}% clients "
                  f"have accuracy >= {self.accuracy_threshold}%")
            # Set flag to complete training
            self.set_flag(self.FLAG.TRAINING_COMPLETE)
        else:
            print(f"Training will continue to next round. Only {percentage_meeting_threshold*100:.1f}% "
                  f"clients met the accuracy threshold (target: {self.completion_threshold*100:.1f}%)")
            # Reset accuracy tracking for next round
            self.client_accuracies = {}
            # Set flag to start a new round
            self.set_flag(self.FLAG.START_ROUND)

class Round_Manager():

    def __init__(self, client_list: list[Client_info], round_number: int):
        self.client_list = client_list
        self.round_number = round_number
        
        # Create graph and add round information for clients
        # Please insert here to specify the neighbor_num more useful

        for round_ID in range(len(self.client_list)):
            self.client_list[round_ID].set_round_information(round_ID)

    def __get_client_by_ID__(self, client_ID: int) -> Client_info | None:
        for client_info in self.client_list:
            if client_info.ID == client_ID:
                return client_info
        return None
    
    def __get_client_by_round_ID__(self, client_round_ID: int) -> Client_info | None:
        for client_info in self.client_list:
            if client_info.round_ID == client_round_ID:
                return client_info
        return None
