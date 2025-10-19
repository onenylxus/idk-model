from termcolor import colored

def print_info(message: str) -> None:
    """Print an informational message to the console."""

    print(colored(' INFO ', 'white', 'on_cyan', ['bold']) + ' ' + colored(message, 'cyan'))

def print_pass(message: str) -> None:
    """Print a success message to the console."""

    print(colored(' PASS ', 'white', 'on_green', ['bold']) + ' ' + colored(message, 'green'))

def print_warn(message: str) -> None:
    """Print a warning message to the console."""

    print(colored(' WARN ', 'white', 'on_yellow', ['bold']) + ' ' + colored(message, 'yellow'))

def print_fail(message: str) -> None:
    """Print a failure message to the console."""

    print(colored(' FAIL ', 'white', 'on_red', ['bold']) + ' ' + colored(message, 'red'))

def print_prompt(message: str) -> None:
    """Print a prompt message to the console."""

    print('\n' + colored('PROMPT: ', 'cyan', None, ['bold']) + ' ' + colored(message, 'cyan'))

def print_response(message: str) -> None:
    """Print a response message to the console."""

    print('\n' + colored('RESPONSE: ', None, None, ['bold']) + ' ' + message)
