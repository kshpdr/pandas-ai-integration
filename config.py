class Config:
    def __init__(self) -> None:
        self.open_ai_key = self.load_env_variable("OPENAPI_KEY")

    def load_env_variable(self, key: str) -> str:
        try:
            with open('.env', 'r') as file:
                for line in file.readlines():
                    if line.startswith(key):
                        return line.split('=')[1].strip()
        except FileNotFoundError:
            print(f"Warning: .env file not found. {key} will be set to an empty string.")
        return ""

    def get_open_ai_key(self):
        return self.open_ai_key
    