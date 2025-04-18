import torch, numpy, os
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from Thread.Worker.BaseModel import *
import torchvision, torch.optim as optim
from tqdm import tqdm
from Thread.Worker.Helper import Helper

class Trainer:

    def __init__(self, model_type: type):
        self.device = Helper.get_device()
        print(f"Trainer initialized with device: {self.device}")
        self.local_model : CNNModel_MNIST = model_type()
        self.dataset_type = model_type.__name__.split('_')[-1]
        self.batch_size = 64
        self.epoch_num = 3
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=0.01, momentum=0.5)
        self.get_parameters()
        # Move model back to CPU after initialization to save GPU memory
        if self.device.type == 'cuda':
            self.local_model.cpu()
            Helper.clear_gpu_memory()
    
    def set_dataset_ID(self, ID: int, round_number: int):
        self.ID = ID

        # Dataset
            
            # Root dataset
        self.root_dataset : type = getattr(torchvision.datasets, self.dataset_type)
        self.root_train_data : torchvision.datasets.MNIST = self.root_dataset(root="Thread/Worker/Data", train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
        self.root_test_data : torchvision.datasets.MNIST = self.root_dataset(root="Thread/Worker/Data", train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
            
            # Self dataset
        ATTEND_CLIENTS = int(Helper.get_env_variable('ATTEND_CLIENTS'))
        # self.data_num = self.root_train_data.__len__() // ATTEND_CLIENTS
        # self.self_train_data = Subset(self.root_train_data, range(self.ID * self.data_num, (self.ID + 1) * self.data_num))

        self.data_num = self.root_train_data.__len__() // 100
        self.self_train_data = Subset(self.root_train_data, range(((round_number * ATTEND_CLIENTS + self.ID) * self.data_num) % self.root_train_data.__len__(), ((round_number * ATTEND_CLIENTS + self.ID + 1) * self.data_num) % self.root_train_data.__len__()))


        # self.test_data_num = self.root_test_data.__len__() // ATTEND_CLIENTS
        # self.self_test_data = Subset(self.root_test_data, range(self.ID * self.test_data_num, (self.ID + 1) * self.test_data_num))

        self.test_data_num = self.root_test_data.__len__() // 100
        self.self_test_data = Subset(self.root_test_data, range(((round_number * ATTEND_CLIENTS + self.ID) * self.test_data_num) % self.root_train_data.__len__(), ((round_number * ATTEND_CLIENTS + self.ID + 1) * self.test_data_num) % self.root_train_data.__len__()))

    @Helper.timing
    def load_parameters(self, parameters: numpy.ndarray[numpy.float32], round_ID: int):
        # Create Models directory if it doesn't exist
        models_dir = "Thread/Worker/Data/Models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Keep model on CPU for parameter loading
        self.local_model.cpu()    
        tensor = torch.tensor(parameters, dtype=torch.float32, requires_grad=True)
        torch.save(self.local_model, f"{models_dir}/{round_ID}_old.pth")
        vector_to_parameters(tensor, self.local_model.parameters())
        torch.save(self.local_model, f"{models_dir}/{round_ID}_new.pth")

    def get_parameters(self) -> numpy.ndarray[numpy.float32]:
        # Make sure model is on CPU for parameter extraction
        self.local_model.cpu()
        return parameters_to_vector(self.local_model.parameters()).detach().numpy()
        
    def __get_data__(self, data: Subset) -> TensorDataset:
        
        if self.root_dataset == torchvision.datasets.MNIST:
            origin_data = torch.stack([data.dataset[idx][0] for idx in data.indices])
            target_label = torch.tensor([data.dataset[idx][1] for idx in data.indices])
            return TensorDataset(origin_data, target_label)
    
        # Please input here any another root_dataset type (cifar10, cifar100, etc.)
        # elif self.root_dataset == torchvision.datasets.CIFAR10:
        #     pass

        else:
            raise Exception("There is no data available to get!")
        
    def __get_train_data__(self) -> TensorDataset:
        return self.__get_data__(self.self_train_data)
    
    def __get_test_data__(self) -> TensorDataset:
        return self.__get_data__(self.self_test_data)

    @Helper.timing
    def train(self):
        train_loader = DataLoader(self.__get_train_data__(), batch_size = self.batch_size)

        # Model will be moved to GPU during forward pass if needed
        for epoch_idx in range(self.epoch_num):
            running_loss = 0.0
            correct = 0
            total = 0

            for data, target in tqdm(train_loader, unit=" data", leave=False):
                # Data will be moved to device in the forward pass
                self.optimizer.zero_grad()
                output = self.local_model(data) # Model and data moved to device here
                loss = F.nll_loss(output, target.to(self.device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target.to(self.device)).sum().item()
                total += target.size(0)

            epoch_loss = running_loss / total
            print("Loss: ", epoch_loss)
            epoch_acc = correct / total
            print(f"Epoch [{epoch_idx+1}/{self.epoch_num}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Move model back to CPU and clear GPU memory after training
        if self.device.type == 'cuda':
            self.local_model.cpu()
            Helper.clear_gpu_memory()

    @torch.no_grad()
    def test(self):
        test_loader = DataLoader(self.__get_test_data__())
        self.local_model.eval()

        test_loss = 0
        correct = 0

        for data, target in tqdm(test_loader, unit=" data", leave=False):
            # Model and data will be moved to device during forward pass
            output = self.local_model(data)
            test_loss += F.nll_loss(output, target.to(self.device), reduction="sum").item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.to(self.device).view_as(pred)).long().cpu().sum()
            
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'[Evaluation]: Test Loss = {test_loss:.4f}, Accuracy = {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
        
        # Move model back to CPU and clear GPU memory after testing
        if self.device.type == 'cuda':
            self.local_model.cpu()
            Helper.clear_gpu_memory()
        
        return float(accuracy)