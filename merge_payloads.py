from myutils.payload_merger import merge_payloads
from myutils.logger_setter import set_logger
from myutils.ConnectionInfo import ConnectionInfo


def run():
    connection_info = ConnectionInfo()

    set_logger(connection_info.team_name)

    merge_payloads(input("First Payload folder: "), input("Second Payload folder: "))


if __name__ == '__main__':
    run()
