from Thread.Worker.Manager import Manager, Round_Manager
from Thread.Worker.Thread_Controller import *
from time import sleep

ATTEND_CLIENTS = Helper.get_env_variable('ATTEND_CLIENTS')

def controller_thread(manager: Manager):

    print("Controller is on and at duty!")
    stop_rounds = False  # Flag to indicate convergence has been reached

    # Get next command from stdinput
    while True:
        
        flag = manager.get_flag()

        if flag == manager.FLAG.STOP:
            print("Got the ABORT signal, send the STOP signal...")
            asyncio.run(send_STOP(manager))

        # Init the round
        elif flag == manager.FLAG.START_ROUND:
            
            manager.round_manager = Round_Manager(manager.choose_clients(ATTEND_CLIENTS), manager.get_current_round())
            asyncio.run(send_ROUND_INFO_client(manager))
            asyncio.run(send_ROUND_INFO_aggregator(manager))

            # Increment the round number after starting a new round
            manager.current_round += 1

        else:
            pass

        sleep(5)