import asyncio, telnetlib3, struct, numpy
from Thread.Worker.Manager import Manager, Client_info
from Thread.Worker.Helper import Helper
from Thread.Worker.Thread_Controller import send_MODEL_ACCURACY

def listener_thread(manager: Manager):
    
    print(f"Listener is on at port {manager.port}")

    async def shell(reader: telnetlib3.TelnetReader, writer: telnetlib3.TelnetWriter):

        data = await Helper.receive_data(reader)

        # Aggregator/Client aborts the process due to abnormal activities
        if b'STOP' == data[:4]:

            message = data[5:].split(b' ', 1)
            print("STOP due to " + message.decode())
            manager.set_flag(manager.FLAG.STOP)

        # Trusted Party gets DH public keys from chosen Clients

        # Trusted Party sends round information to Clients
        elif b'ROUND_INFO' == data[:10]:

            # ROUND_INFO <round_number> <client_round_ID> 
            round_number, self_round_ID = data[11:].split(b' ', 1)
            round_number, self_round_ID = int(round_number), int(self_round_ID)
            
            # <base_model_commit/previous_global_model_commit>
            # data = await Helper.receive_data(reader)
            # manager.set_last_commit(numpy.frombuffer(data, dtype=numpy.uint64))
            # print("Confirm to get the model commit from the Trusted party")

            
            manager.set_round_information(round_number, self_round_ID)

            # SUCCESS
            await Helper.send_data(writer, "SUCCESS")
            print("Successfully receive round information from the Trusted party")
            writer.close()

        # Aggregator sends global model to Clients
        elif b'GLOB_MODEL' == data[:10]:

            # <global_model_parameters>
            data = await Helper.receive_data(reader)
            
            # print(f"Get global parameters for the round {manager.round_number}")
            global_parameters = numpy.frombuffer(data, dtype=numpy.float32)
            manager.trainer.load_parameters(global_parameters, manager.round_ID)
            # SUCCESS
            await Helper.send_data(writer, "SUCCESS")
            print("Successfully receive global model from the Aggregator")
            manager.set_flag(manager.FLAG.TRAIN)
            
            writer.close()

        # Client sends secret points to its neighbors

        # Aggregator gets secrets points from Clients
    
        # Aggregator sends aggregated global model to Clients
        elif data[:9] == b'AGG_MODEL':

            # <global_parameters>
            data = await Helper.receive_data(reader)
            received_global_parameters = numpy.frombuffer(data, dtype=numpy.float32)

            manager.trainer.load_parameters(received_global_parameters, manager.round_ID)
            
            await Helper.send_data(writer, "SUCCESS")
            print(f"Successfully receive global models from the Aggregator")
            writer.close()
            
            # Test the model and send accuracy to Trusted Party
            asyncio.ensure_future(send_MODEL_ACCURACY(manager))
        else:
            await Helper.send_data(writer, "Operation not allowed!")
        
        writer.close()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    coro = telnetlib3.create_server(port=manager.port, shell=shell, encoding=False, encoding_errors="ignore")
    server = loop.run_until_complete(coro)
    loop.run_until_complete(server.wait_closed())