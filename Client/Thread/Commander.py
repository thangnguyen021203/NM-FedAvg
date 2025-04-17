from Thread.Worker.Manager import Manager
import os, sys

def commander_thread(manager: Manager):

    print("Commander is on and wait for input!")

    while True:

        command = input()

        if command == 'stop':
            print("Client stops!")
            quit()

        elif command == "client info":
            print(f"Self info: {manager.host}:{manager.port}")
            print(f"Aggregator info - {manager.aggregator_info.host}:{manager.aggregator_info.port}")

        elif command == "round info":
            if manager.round_ID is None:
                print("Client is currently not in training round")
                continue
            print(f"ID: {manager.round_ID}")
        

        elif command == 'register':
            manager.set_flag(manager.FLAG.RE_REGISTER)

        elif command == 'cls':
            os.system('cls')
        
        elif command == 'restart':
            os.execv(sys.executable, ['python'] + sys.argv)

        elif command[:5] == 'abort':
            manager.abort_message == command[6:]
            manager.set_flag(manager.FLAG.ABORT)

        elif command == 'test model':
            manager.trainer.test_model()

        else:
            print("I'm currently supporting these commands: [stop, client info, round info, register, cls, restart, abort <message>, test model]")