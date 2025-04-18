from Thread.Worker.Manager import Manager
import os, sys

def commander_thread(manager: Manager):
    print("Commander is on and waiting for input!")

    while True:
        command = input()

        if command == 'stop':
            print("Aggregator stop!")
            manager.set_flag(manager.FLAG.STOP)
            quit()
        
        elif command == 'round info':
            print("Attendee clients in round:")
            for client in manager.client_list:
                print(f"Client {client.client_id} - {client.host}:{client.port}")
                print(f"Dataset size: {client.data_num}")
                print("---")

        elif command == 'info':
            print(f"Aggregator address: {manager.host}:{manager.port}")
            print(f"Round number: {manager.round_number}")
            print(f"Clients connected: {len(manager.client_list)}")
            print(f"Timeout status: {'Timed out' if manager.timeout else 'Collecting updates'}")

        elif command == 'start':
            print(f"Starting round {manager.round_number + 1}")
            manager.round_number += 1
            manager.set_flag(manager.FLAG.START_ROUND)
        
        elif command == 'aggregate':
            print("Manually triggering aggregation")
            manager.set_flag(manager.FLAG.AGGREGATE)
        
        elif command == 'cls':
            # os.system('cls')
            os.system('clear' if os.name != 'nt' else 'cls')
        
        elif command == 'restart':
            os.execv(sys.executable, ['python'] + sys.argv)

        elif command == 'timeout':
            if manager.timeout:
                print("Collection period has ended")
            else:
                print(f"Collecting updates. Received: {manager.received_data}/{len(manager.client_list)}")

        elif command == 'cancel':
            print("Canceling timer and ending collection period")
            if hasattr(manager, 'timer'):
                manager.timer.cancel()
            manager.end_timer()

        else:
            print("Available commands: [stop, clients, info, start, aggregate, cls, restart, timeout, cancel]")