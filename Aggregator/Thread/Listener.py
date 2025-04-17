import asyncio, telnetlib3, dill as pickle, numpy
from Thread.Worker.Manager import Manager, Client_info
from Thread.Worker.Helper import Helper

def listener_thread(manager: Manager):

    print(f"Listener is on at port {manager.port}")

    async def shell(reader: telnetlib3.TelnetReader, writer: telnetlib3.TelnetWriter):
            
        data = await Helper.receive_data(reader)
        
        # Aggregator/Client aborts the process due to abnormal activities
        if b'STOP' == data[:4]:

            message = data[5:].split(b' ', 1)
            print("STOP due to " + message.decode())
            manager.set_flag(manager.FLAG.STOP)

        # Trusted Party sends round information to Aggregator
        elif b'ROUND_INFO' == data[:10]:

            # ROUND_INFO <round_number> <client_num>
            manager.round_number, client_num = [int(x) for x in data[11:].split(b' ', 1)]
            client_list = list()
            for i in range(client_num):

                # <client_round_ID> <client_host> <client_port>
                data : bytes = await Helper.receive_data(reader)
                round_ID, host, port = data.split(b' ', 2)
                host = host.decode()
                round_ID, port = [int(param) for param in [round_ID, port]]

                client_list.append(Client_info(round_ID, host, port))
                print(f"Successfully receive information of client {round_ID}")

            # SUCCESS
            await asyncio.wait_for(Helper.send_data(writer, "SUCCESS"), timeout=None)
            manager.set_round_information(client_list)
            manager.set_flag(manager.FLAG.START_ROUND)
        
        # Client sends local model to Aggregator
        elif b'LOCAL_MODEL' == data[:11]:
            
            # LOCAL_MODEL <round_ID> <data_number>
            client_round_ID, data_number = data[12:].split(b' ', 1)
            client_round_ID, data_number = int(client_round_ID), int(data_number)
            # print(f"Get local model information from client {round_ID}")

            # <local_model_parameters>
            data: bytes = await Helper.receive_data(reader)
            local_model_parameters = numpy.frombuffer(data, dtype=numpy.float32)  # Explicitly specify float32
            
            # Debug prints to verify parameter size
            print(f"Received {len(local_model_parameters)} parameters from client {client_round_ID}")
            
            if not manager.timeout:
                client = manager.get_client_by_id(client_round_ID)
                
                # Test the received model using the client's test dataset
                print(f"Testing received model from client {client_round_ID}...")
                accuracy = manager.test_client_model(client_round_ID, local_model_parameters)
                
                # Store the model parameters
                manager.receive_trained_data(client, data_number, local_model_parameters)
                manager.received_data += 1
                
                data = f"SUCCESS"
                await Helper.send_data(writer, data)
                print(f"Successfully receive local model parameters of client {client_round_ID}")

            else:

                # OUT_OF_TIME <end_time>
                data = f"OUT_OF_TIME {manager.timeout_time}"
                await Helper.send_data(writer, data)
                print(f"Client {client_round_ID} has been late for the timeout of {manager.timeout_time}!")

        # Trusted Party requests testing of the aggregated model
        elif b'TEST_MODEL' == data:
            print("Received request to test aggregated model from Trusted Party")
            
            # Only test if we have a global model
            if manager.global_model is not None:
                # Test the model on the whole dataset
                accuracy = manager.test_aggregated_model_on_whole_dataset()
                
                # Send accuracy result back to Trusted Party
                await Helper.send_data(writer, f"ACCURACY {accuracy}")
                print(f"Sent model accuracy ({accuracy:.2f}%) to Trusted Party")
            else:
                await Helper.send_data(writer, "ERROR No global model available")
                print("Error: No global model available for testing")

        else:
            await Helper.send_data(writer, "Operation not allowed!")
        
        writer.close()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    coro = telnetlib3.create_server(port=manager.port, shell=shell, encoding=False, encoding_errors='ignore')
    server = loop.run_until_complete(coro)
    loop.run_until_complete(server.wait_closed())