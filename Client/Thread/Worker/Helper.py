from hashlib import sha256
from Crypto.Cipher import AES
import json, time, random, asyncio, telnetlib3
from socket import socket, AF_INET, SOCK_STREAM
import torch

class Helper:

    @staticmethod
    def timing(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function {func.__name__} executed in {end_time - start_time:.6} seconds")
            return result
        return wrapper

    @staticmethod
    def exponent_modulo(base: int, exponent: int, modulo: int) -> int:
        if modulo == 1:
            return 0
        
        result = 1
        while exponent > 0: 
            if exponent % 2:
                base, exponent, result = (base * base) % modulo, exponent // 2, (base * result) % modulo
            else:
                base, exponent, result = (base * base) % modulo, exponent // 2, result
        return result
    
    @staticmethod
    def PRNG(seed: int, num_bytes: int) -> int:
        key = sha256(str(seed).encode()).digest()[:16]
        cipher = AES.new(key, AES.MODE_CTR, nonce = key[:12])
        random_bytes = cipher.encrypt(b'\x00' * num_bytes)
        return int.from_bytes(random_bytes, "big", signed=True)
    
    @staticmethod
    def get_device():
        """Return the best available device based on env setting (CUDA GPU if available and enabled, otherwise CPU)"""
        try:
            use_gpu = Helper.get_env_variable("USE_GPU")
            if use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"GPU device available: {torch.cuda.get_device_name(0)}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
                return device
            else:
                return torch.device("cpu")
        except (KeyError, FileNotFoundError):
            # Fall back to CPU if setting not found or file not accessible
            return torch.device("cpu")
        
    @staticmethod
    def get_env_variable(name: str) -> int | str:
        return json.load(open("../env.env", "r", encoding='UTF-8'))[name]
    
    @staticmethod
    def get_available_port() -> int:
        while True:
            port = random.randint(30000, 60000)
            if socket(AF_INET, SOCK_STREAM).connect_ex(('localhost', port)):
                return port
    
    @staticmethod
    async def send_data(writer: asyncio.StreamWriter | telnetlib3.TelnetWriter, data: str | bytes) -> None:
        
        if type(data) == str:
            data = data.encode()

        # Send data_len
        data_len = len(data)
        writer.write(f"{str(data_len)}\n".encode())
        await writer.drain()

        # Send data
        writer.write(data.replace(b'\xff', b'\xff\xff'))
        await writer.drain()

    @staticmethod
    async def receive_data(reader: asyncio.StreamReader | telnetlib3.TelnetReader) -> bytes:
        try:
            # Receive data_len
            data_len = await reader.readuntil()
            data_len = int(data_len)

            # Receive data
            data = await reader.readexactly(data_len)
            return data
        except asyncio.IncompleteReadError as e:
            print(f"Connection closed unexpectedly. Received {len(e.partial)} bytes.")
            return e.partial if e.partial else b""
        except (ConnectionResetError, ConnectionAbortedError) as e:
            print(f"Connection error: {e}")
            return b""

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache to free up resources"""
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            