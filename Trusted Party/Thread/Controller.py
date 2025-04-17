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
            if stop_rounds:
                print("Convergence criteria met - no more rounds will be initiated")
                sleep(10)
                continue
            
            # Start a new round
            manager.current_round += 1
            print(f"\n===== Starting Round {manager.current_round} =====")
            
            manager.round_manager = Round_Manager(manager.choose_clients(ATTEND_CLIENTS), manager.get_current_round())
            asyncio.run(send_ROUND_INFO_client(manager))
            asyncio.run(send_ROUND_INFO_aggregator(manager))
            
            # Wait for the round to complete and aggregation to happen
            # We use a fixed wait time - a real implementation would use signals or callbacks
            sleep(120)  # Wait 2 minutes for the round to complete
            
            # Test the aggregated model and check for convergence
            print("\nTesting global model after round completion...")
            accuracy = manager.test_aggregated_model()
            
            # Check if we've reached convergence
            if accuracy > 0 and manager.check_convergence(accuracy):
                print("Convergence criteria met - no more rounds will be initiated")
                stop_rounds = True
            else:
                print(f"Convergence criteria not met - will initiate another round")
                # Set flag to start next round immediately
                manager.set_flag(manager.FLAG.START_ROUND)

        else:
            pass

        sleep(5)