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
            return parameters_to_vector(self.global_model.parameters()).detach().numpy()
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
        
        total_parameters = [0 for _ in range(len(self.client_list[0].local_parameters))]

        for client in self.client_list:
            print(f"10 gia tri dau {client}",client.local_parameters[:10])
            for idx in range(len(total_parameters)):
                total_parameters[idx] += client.local_parameters[idx]*client.local_datanum
        print("10 gia tri dau sau khi tong hop: ", total_parameters[:10])

        total_data_num = sum([client.local_datanum for client in self.client_list])
        self.global_parameters = [param/total_data_num for param in total_parameters]
        params_tensor = torch.tensor(self.global_parameters, dtype=torch.float32)
        old_parameter = self.global_model.parameters()
        vector_to_parameters(params_tensor, self.global_model.parameters())
        print("PTV: ",parameters_to_vector(self.global_model.parameters()))

        # Cái này để debug
        new_parameter = self.global_model.parameters()
        if old_parameter != new_parameter:
            print("Parameter is diffirent! It's OK!!!")
        print("\n===== Testing Aggregated Model =====")
        self.test_aggregated_model_on_client_datasets()
        self.test_aggregated_model_on_whole_dataset()
        print("===================================\n")

    def test_client_model(self, client_id: int, parameters: numpy.ndarray) -> float:
        """Check coi nhận model từ mỗi thằng client có chính xác không"""
        import torch
        import torch.nn.functional as F
        import torchvision
        from torch.utils.data import DataLoader, TensorDataset, Subset
        from tqdm import tqdm
        
        # Create a temporary model from the received parameters
        temp_model = self.model_type()
        temp_tensor = torch.tensor(parameters, dtype=torch.float32, requires_grad=False)
        vector_to_parameters(temp_tensor, temp_model.parameters())
        temp_model.eval()
        
        # Get test dataset for verification
        try:
            dataset_type = self.model_type.__name__.split('_')[-1]  # Extract dataset type from model name
            root_dataset = getattr(torchvision.datasets, dataset_type)
            test_data = root_dataset(root="Thread/Worker/Data", train=False, download=True, 
                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
            
            # Create client-specific test data subset
            ATTEND_CLIENTS = int(Helper.get_env_variable('ATTEND_CLIENTS'))
            test_data_num = test_data.__len__() // 10
            test_indices = range((self.round_number * ATTEND_CLIENTS + client_id) * test_data_num, 
                                (self.round_number * ATTEND_CLIENTS + client_id + 1) * test_data_num)
            client_test_data = Subset(test_data, test_indices)
            
            # Prepare test data
            def get_test_data(data_subset):
                origin_data = torch.stack([data_subset.dataset[idx][0] for idx in data_subset.indices])
                target_label = torch.tensor([data_subset.dataset[idx][1] for idx in data_subset.indices])
                return TensorDataset(origin_data, target_label)
            
            test_dataset = get_test_data(client_test_data)
            test_loader = DataLoader(test_dataset, batch_size=64)
            
            # Evaluate model
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in tqdm(test_loader, unit=" data", leave=False, desc=f"Testing Client {client_id} Model"):
                    output = temp_model(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).long().cpu().sum().item()
                    total += target.size(0)
            
            if total > 0:
                test_loss /= total
                accuracy = 100.0 * correct / total
                print(f'[Client {client_id} Model Test]: Loss={test_loss:.4f}, Accuracy={correct}/{total} ({accuracy:.1f}%)')
                return accuracy
            else:
                print(f"Warning: No test data for client {client_id}")
                return 0.0
                
        except Exception as e:
            print(f"Error testing client model: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
            
    def test_aggregated_model_on_client_datasets(self) -> float:
        """Check coi sau khi tổng hợp thì với mỗi tập test_dataset nó có bị giảm đáng kể không"""
        import torch
        import torch.nn.functional as F
        import torchvision
        from torch.utils.data import DataLoader, TensorDataset, Subset
        from tqdm import tqdm
        
        if self.global_model is None:
            print("Error: No global model available for testing")
            return 0.0
            
        self.global_model.eval()
        
        try:
            # Get test dataset
            dataset_type = self.model_type.__name__.split('_')[-1]
            root_dataset = getattr(torchvision.datasets, dataset_type)
            test_data = root_dataset(root="Thread/Worker/Data", train=False, download=True, 
                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
            
            # Testing on each client's test dataset
            overall_correct = 0
            overall_total = 0
            
            print("\n----- Testing Aggregated Model on Each Client's Test Dataset -----")
            
            for client in self.client_list:
                client_id = client.round_ID
                
                # Create client-specific test data subset
                ATTEND_CLIENTS = int(Helper.get_env_variable('ATTEND_CLIENTS'))
                test_data_num = test_data.__len__() // 10
                test_indices = range((self.round_number * ATTEND_CLIENTS + client_id) * test_data_num, 
                                    (self.round_number * ATTEND_CLIENTS + client_id + 1) * test_data_num)
                client_test_data = Subset(test_data, test_indices)
                
                # Prepare test data
                def get_test_data(data_subset):
                    origin_data = torch.stack([data_subset.dataset[idx][0] for idx in data_subset.indices])
                    target_label = torch.tensor([data_subset.dataset[idx][1] for idx in data_subset.indices])
                    return TensorDataset(origin_data, target_label)
                
                test_dataset = get_test_data(client_test_data)
                test_loader = DataLoader(test_dataset, batch_size=64)
                
                # Evaluate model
                test_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in tqdm(test_loader, unit=" data", leave=False, 
                                            desc=f"Testing on Client {client_id} Test Data"):
                        output = self.global_model(data)
                        test_loss += F.nll_loss(output, target, reduction="sum").item()
                        pred = output.max(1, keepdim=True)[1]
                        correct += pred.eq(target.view_as(pred)).long().cpu().sum().item()
                        total += target.size(0)
                
                if total > 0:
                    test_loss /= total
                    accuracy = 100.0 * correct / total
                    print(f'[Client {client_id} Test Data]: Loss={test_loss:.4f}, Accuracy={correct}/{total} ({accuracy:.1f}%)')
                    overall_correct += correct
                    overall_total += total
            
            # Calculate overall accuracy across clients
            if overall_total > 0:
                overall_accuracy = 100.0 * overall_correct / overall_total
                print(f'\n[Overall Client Test Results]: Accuracy={overall_correct}/{overall_total} ({overall_accuracy:.1f}%)')
                return overall_accuracy
            else:
                print("Warning: No client test data available")
                return 0.0
                
        except Exception as e:
            print(f"Error testing aggregated model on client datasets: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
            
    def test_aggregated_model_on_whole_dataset(self) -> float:
        """Check trên toàn bộ testdataset trước khi gửi cho client"""
        import torch
        import torch.nn.functional as F
        import torchvision
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        if self.global_model is None:
            print("Error: No global model available for testing")
            return 0.0
            
        self.global_model.eval()
        
        try:
            # Get test dataset
            dataset_type = self.model_type.__name__.split('_')[-1]
            root_dataset = getattr(torchvision.datasets, dataset_type)
            test_data = root_dataset(root="Thread/Worker/Data", train=False, download=True, 
                                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
            
            print("\n----- Testing Aggregated Model on Full Test Dataset -----")
            
            # Create data loader for whole test dataset
            test_loader = DataLoader(test_data, batch_size=128)
            
            # Evaluate model
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in tqdm(test_loader, unit=" batch", leave=False, 
                                        desc=f"Testing on Full Test Dataset"):
                    output = self.global_model(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).long().cpu().sum().item()
                    total += target.size(0)
            
            if total > 0:
                test_loss /= total
                accuracy = 100.0 * correct / total
                print(f'[Full Test Dataset]: Loss={test_loss:.4f}, Accuracy={correct}/{total} ({accuracy:.1f}%)')
                return accuracy
            else:
                print("Warning: No test data available")
                return 0.0
                
        except Exception as e:
            print(f"Error testing aggregated model on whole dataset: {e}")
            import traceback
            traceback.print_exc()
            return 0.0