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
        # Model performance tracking
        self.last_accuracy = 0.0
        self.convergence_threshold = 0.05  # 0.05% improvement threshold
        self.target_accuracy = 90.0  # 90% accuracy target

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
    
    def test_aggregated_model(self) -> float:
        """
        Request the aggregator to test the global model on the whole test dataset
        and return the accuracy for convergence checking
        """
        import asyncio
        from Thread.Worker.Helper import Helper
        
        async def request_model_test():
            try:
                # Connect to aggregator
                reader, writer = await asyncio.open_connection(
                    self.aggregator_info.host, 
                    self.aggregator_info.port
                )
                _ = await reader.read(3)  # Remove first 3 bytes of Telnet command
                
                # Send TEST_MODEL command
                await Helper.send_data(writer, "TEST_MODEL")
                
                # Receive test results
                data = await Helper.receive_data(reader)
                if data.startswith(b"ACCURACY "):
                    accuracy = float(data[9:])
                    print(f"Global model accuracy: {accuracy:.2f}%")
                    writer.close()
                    return accuracy
                else:
                    print(f"Unexpected response from aggregator: {data}")
                    writer.close()
                    return 0.0
                    
            except Exception as e:
                print(f"Error testing aggregated model: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
        
        # Run the async function and return the result
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        accuracy = loop.run_until_complete(request_model_test())
        loop.close()
        return accuracy
    
    def check_convergence(self, current_accuracy: float) -> bool:
        """
        Check if the model has converged based on:
        1. Accuracy exceeding target (90%)
        2. Improvement less than threshold (0.05%)
        """
        # Calculate accuracy improvement from previous round
        accuracy_improvement = current_accuracy - self.last_accuracy
        
        print(f"Current accuracy: {current_accuracy:.2f}%, Previous: {self.last_accuracy:.2f}%")
        print(f"Improvement: {accuracy_improvement:.3f}%")
        
        # Update the last_accuracy for the next round
        self.last_accuracy = current_accuracy
        
        # Check if we've reached our convergence criteria
        if current_accuracy >= self.target_accuracy and accuracy_improvement < self.convergence_threshold:
            print(f"\n===== CONVERGENCE REACHED =====")
            print(f"Accuracy: {current_accuracy:.2f}% (Target: {self.target_accuracy:.2f}%)")
            print(f"Improvement: {accuracy_improvement:.3f}% (Threshold: {self.convergence_threshold:.3f}%)")
            print(f"Stopping federated learning process")
            print(f"================================\n")
            return True
        
        return False

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
