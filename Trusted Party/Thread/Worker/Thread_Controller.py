import asyncio, struct
from Thread.Worker.Helper import Helper
from Thread.Worker.Manager import Manager, Client_info


# Aggregator/Client aborts the process due to abnormal activities
async def send_STOP(manager: Manager):

    # STOP <message>
    if not manager.stop_message:
        manager.stop_message = "No message specified"

    for client in manager.client_list:
        reader, writer = await asyncio.open_connection(client.host, client.port)
        _ = await reader.read(3)  # Remove first 3 bytes of Telnet command
        await Helper.send_data(writer, f"STOP {manager.current_round} {manager.stop_message}")
        writer.close()

    reader, writer = await asyncio.open_connection(manager.aggregator_info.host, manager.aggregator_info.port)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command
    await Helper.send_data(writer, f"STOP {manager.current_round} {manager.stop_message}")
    writer.close()



###########################################################################################################



# Trusted Party gets DH public keys from chosen Clients




###########################################################################################################



# Trusted Party sends round information to Clients
async def send_ROUND_INFO_client_each(manager: Manager, client: Client_info):

    reader, writer = await asyncio.open_connection(client.host, client.port)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command

    # ROUND_INFO <round_number> <client_round_ID>
    data = f'ROUND_INFO {manager.round_manager.round_number} {client.round_ID}'
    await Helper.send_data(writer, data)

    # <base_model_commit/previous_global_model_commit>
    # data = manager.last_commitment.tobytes()
    # await Helper.send_data(writer, data)
    
    # SUCCESS
    data = await Helper.receive_data(reader)
    if data == b"SUCCESS":
        print(f"Successfully send round information to client {client.round_ID}")
    else:
        print(f"Client {client.round_ID} returns {data}")
    writer.close()

async def send_ROUND_INFO_client(manager: Manager):

    for client in manager.round_manager.client_list:
        asyncio.create_task(send_ROUND_INFO_client_each(manager, client))
    all_remaining_tasks = asyncio.all_tasks()
    all_remaining_tasks.remove(asyncio.current_task())
    await asyncio.wait(all_remaining_tasks)



###########################################################################################################



# Trusted Party sends round information to Aggregator
async def send_ROUND_INFO_aggregator(manager: Manager):

    reader, writer = await asyncio.open_connection(manager.aggregator_info.host, manager.aggregator_info.port)
    _ = await reader.read(3)  # Remove first 3 bytes of Telnet command

    # ROUND_INFO <round_number> <client_num>
    data = f"ROUND_INFO {manager.round_manager.round_number} {len(manager.round_manager.client_list)}"
    await Helper.send_data(writer, data)

    for client in manager.round_manager.client_list:

        # <client_round_ID> <client_host> <client_port>
        data = f"{client.round_ID} {client.host} {client.port}"
        await Helper.send_data(writer, data)
    
    # SUCCESS
    data = await Helper.receive_data(reader)
    if data == b"SUCCESS":
        print(f"Successfully send round information to the Aggregator")
    else:
        print(f"Aggregator returns {data}")
    writer.close()