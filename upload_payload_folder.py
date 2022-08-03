from src.connection_handler import ConnectionHandler

from myutils.logger_setter import set_logger
from myutils.ConnectionInfo import ConnectionInfo


def run():
    print("Started...")
    # Get configurations from .env file
    payload_folder = input("Payload Folder Path: \n")

    connection_info = ConnectionInfo()

    # Declare logging configuration.
    # configure_logger(team_name)
    set_logger(connection_info.team_name)

    # Connect to the evaluation server.
    server = ConnectionHandler(connection_info.evaluation_server_url,
                               username=connection_info.team_name,
                               password=connection_info.password)

    server.upload_payloads(payload_folder)


if __name__ == '__main__':
    run()
