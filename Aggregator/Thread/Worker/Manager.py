from Thread.Worker.BaseModel import *
import numpy, time, threading
from Thread.Worker.Helper import Helper
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch

class Client_info:
    def __init__(self, round_ID: int, host: str, port: int):
        # Client identification
        self.round_ID = round_ID
        self.host = host
        self.port = port
        # Training data
        self.local_parameters = None
        self.local_datanum = 0

    def set_trained_data(self, data_number:int, parameters):
        self.local_datanum = data_number
        self.local_parameters = parameters

class Manager:
    """Simplified Aggregator Manager for FedAvg"""

    class FLAG:
        class NONE:
            # Default value
            pass
        class START_ROUND:
            # When starting a new round
            pass
        class ABORT:
            # Used to send abort signal to Trusted party
            pass
        class STOP:
            # Used to stop processing
            pass
        class AGGREGATE:
            # Used to perform model aggregation
            pass
        
        class RE_REGISTER:
            # When commander wants to re-register
            pass
    
    def __init__(self, model_type: type):
        # Device
        self.device = Helper.get_device()
        print(f"Aggregator Manager initialized with device: {self.device}")
        
        # Communication
        self.host = "localhost"
        self.port = Helper.get_available_port()
        # Round parameters
        self.round_number = 0
        # Controller
        self.flag = Manager.FLAG.NONE
        self.abort_message = ""
        self.timeout = True
        self.timeout_time = 0
        self.received_data = 0
        # Model
        self.global_model : CNNModel_MNIST | None = model_type()
        self.model_type = model_type
        self.global_parameters = None

        # Client data
        self.client_list = None

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

    def set_round_information(self, client_list):
        """Set round information with simplified client data list"""
        self.client_list = client_list
        self.received_data = 0  # Reset received data counter
        print(f"Starting new round with {len(client_list)} clients. Reset received data counter.")

    def get_global_parameters(self) -> numpy.ndarray[numpy.float32 | numpy.int64]:
        if not self.global_model is None:
            # Move parameters to CPU before converting to numpy array
            return parameters_to_vector(self.global_model.parameters()).detach().cpu().numpy()
        else:
            return self.global_parameters

    def get_client_by_id(self, client_id: int):
        """Get client by ID"""
        for client in self.client_list:
            if client.round_ID == client_id:
                return client
        return None
    
    def abort(self, message: str):
        self.abort_message = message
        self.set_flag(self.FLAG.ABORT)

    def receive_trained_data(self, client: Client_info, data_number: int, parameters: numpy.ndarray) -> None:
        """Receive trained data from a client"""
        client.set_trained_data(data_number, parameters)
        self.received_data += 1
        print(f"Received trained data from client {client.round_ID} ({self.received_data}/{len(self.client_list)} clients)")

    def end_timer(self):
        """End the timer for collecting client updates"""
        self.timeout = True
        self.timeout_time = time.time()
        self.set_flag(self.FLAG.AGGREGATE)
        print(f"Timer ended at {self.timeout_time}")

    def the_checker(self):
        """Check if all clients have submitted their updates"""
        while True:
            if self.timeout:
                return
            elif self.received_data == len(self.client_list):
                print("All clients have sent their data")
                self.timer.cancel()
                self.end_timer()
                return
            time.sleep(1)

    def start_timer(self, timeout_seconds: int = 60):
        """Start a timer for collecting client updates"""
        self.timeout = False
        self.timer = threading.Timer(timeout_seconds, self.end_timer)
        self.timer.start()

        self.checker = threading.Thread(target=self.the_checker)
        self.checker.daemon = True
        self.checker.start()

    @Helper.timing
    def aggregate(self) -> None:
        """Aggregate model parameters from all clients using weighted FedAvg"""
        # Create tensor on CPU first, then move to device
        total_parameters = torch.zeros(len(self.client_list[0].local_parameters), dtype=torch.float32)
        
        for client in self.client_list:
            # Convert numpy parameters to torch tensor and move to device
            client_tensor = torch.tensor(client.local_parameters, dtype=torch.float32)
            total_parameters += client_tensor * client.local_datanum

        total_data_num = sum([client.local_datanum for client in self.client_list])
        # Perform aggregation calculation
        self.global_parameters = (total_parameters / total_data_num).numpy()
        
        # Convert aggregated parameters to tensor and load into model
        params_tensor = torch.tensor(self.global_parameters, dtype=torch.float32, device=self.device)
        vector_to_parameters(params_tensor, self.global_model.parameters())

        print("===================================\n")
