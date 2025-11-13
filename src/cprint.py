from termcolor import colored

def print_info(message):
    """Print an informational message to the console."""

    print(colored(' INFO ', 'white', 'on_cyan', ['bold']) + ' ' + colored(message, 'cyan'))

def print_pass(message):
    """Print a success message to the console."""

    print(colored(' PASS ', 'white', 'on_green', ['bold']) + ' ' + colored(message, 'green'))

def print_warn(message):
    """Print a warning message to the console."""

    print(colored(' WARN ', 'white', 'on_yellow', ['bold']) + ' ' + colored(message, 'yellow'))

def print_fail(message):
    """Print a failure message to the console."""

    print(colored(' FAIL ', 'white', 'on_red', ['bold']) + ' ' + colored(message, 'red'))

def print_prompt(message):
    """Print a prompt message to the console."""

    print('\n' + colored('PROMPT: ', 'cyan', None, ['bold']) + ' ' + colored(message, 'cyan'))

def print_response(message):
    """Print a response message to the console."""

    print('\n' + colored('RESPONSE: ', None, None, ['bold']) + ' ' + message)

def print_answer(message):
    """Print an answer message to the console."""

    print('\n' + colored('ANSWER: ', 'magenta', None, ['bold']) + ' ' + colored(message, 'magenta'))
