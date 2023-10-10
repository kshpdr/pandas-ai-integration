class QueryHistory:
    def __init__(self):
        self._history = []

    def add_query(self, query, response, command):
        self._history.append({'query': query, 'response': response, 'command': command})

    def get_last_query(self):
        return str(self._history[-1]) if self._history else ""

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history

global_query_history = QueryHistory()