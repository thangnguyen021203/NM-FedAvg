import asyncio, dill as pickle, time
from Thread.Worker.Helper import Helper
from Thread.Worker.Manager import Manager, Client_info

TRUSTED_PARTY_HOST = Helper.get_env_variable("TRUSTED_PARTY_HOST")
TRUSTED_PARTY_PORT = Helper.get_env_variable("TRUSTED_PARTY_PORT")

# Client registers itself with Trusted Party
async def send_CLIENT(manager: Manager):

    reader, writer = await asyncio.open_connection(TRUSTED_PARTY_HOST, TRUSTED_PARTY_PORT)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command
    
    # CLIENT <client_host> <client_port> 
    data = f'CLIENT {manager.host} {manager.port}'
    await Helper.send_data(writer, data)
    
    # <aggregator_host> <aggregator_port> 
    data = await Helper.receive_data(reader)
    host, port = data.split(b' ', 1) 
    host = host.decode()
    port = int(port)
    # commiter = Commiter(tuple([int(param) for param in [p, h, k]]))

    # <base_model_class>
    data = await Helper.receive_data(reader)
    base_model_class = pickle.loads(data)
    manager.set_FL_public_params(host, port, base_model_class)

    # SUCCESS
    await Helper.send_data(writer, "SUCCESS")
    print("Successfully register with the Trusted party")
    writer.close()



###########################################################################################################



# Aggregator/Client aborts the process due to abnormal activities
async def send_ABORT(message: str):

    reader, writer = await asyncio.open_connection(TRUSTED_PARTY_HOST, TRUSTED_PARTY_PORT)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command

    # ABORT <message>
    await Helper.send_data(writer, "ABORT " + message)
    writer.close()



###########################################################################################################



# Client sends secret points to its neighbors



###########################################################################################################



async def send_LOCAL_MODEL(manager: Manager):

    reader, writer = await asyncio.open_connection(manager.aggregator_info.host, manager.aggregator_info.port)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command

    # LOCAL_MODEL <round_ID> <data_number>
    data = f"LOCAL_MODEL {manager.round_ID} {manager.trainer.data_num}"
    await Helper.send_data(writer, data)

    # <local_model_parameters>
    local_model_params = manager.get_model()
    # print(f"Local model parameter size: {local_model_params.size}")
    
    data = local_model_params.tobytes()
    await Helper.send_data(writer, data)

    # SUCCESS <received_time>
    data = await Helper.receive_data(reader)
    if data[:7] == b"SUCCESS":
        pass
    
    # OUT_OF_TIME <end_time>
    elif data[:11] == b'OUT_OF_TIME':
        print(f"Aggregator timer ends at {float(data[12:])}, it is {time.time()} now!")

    else:
        print(f"Trusted party returns {data}")
    writer.close()



###########################################################################################################



# Client sends model accuracy to Trusted Party
async def send_MODEL_ACCURACY(manager: Manager):

    reader, writer = await asyncio.open_connection(TRUSTED_PARTY_HOST, TRUSTED_PARTY_PORT)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command

    # Test the aggregated model
    accuracy = manager.test_aggregated_model()
    
    # MODEL_ACCURACY <round_ID> <accuracy>
    data = f"MODEL_ACCURACY {manager.round_ID} {accuracy}"
    await Helper.send_data(writer, data)
    
    # SUCCESS (just an acknowledgment)
    data = await Helper.receive_data(reader)
    if data == b"SUCCESS":
        print(f"Successfully sent model accuracy to the Trusted party")
    else:
        print(f"Trusted party returns {data}")
    writer.close()
