from decouple import config


class ConnectionInfo:
    def __init__(self, config_path="./config/"):
        config.search_path = config_path
        self.team_name = config('TEAM_NAME')
        self.password = config('PASSWORD')
        self.evaluation_server_url = config("EVALUATION_SERVER_URL")